use std::sync::Arc;

use crate::nostr::messages::{create_dlc_msg_event, handle_dlc_msg_event};
use crate::DlcDevKitDlcManager;
use crate::{nostr, Transport};
use crate::{Oracle, Storage};
use bitcoin::bip32::Xpriv;
use bitcoin::Network;
use nostr_rs::{secp256k1::Secp256k1, Keys, Timestamp, Url};
use nostr_sdk::{Client, RelayPoolNotification};
use tokio::sync::watch;
use tokio::task::JoinHandle;

pub struct NostrDlc {
    pub keys: Keys,
    pub relay_url: Url,
    pub client: Client,
}

impl NostrDlc {
    pub async fn new(
        seed_bytes: &[u8; 32],
        relay_host: &str,
        network: Network,
    ) -> anyhow::Result<NostrDlc> {
        tracing::info!("Creating Nostr Dlc handler.");
        let secp = Secp256k1::new();
        let seed = Xpriv::new_master(network, seed_bytes)?;
        let keys = Keys::new_with_ctx(&secp, seed.private_key.into());

        let relay_url = relay_host.parse()?;
        let client = Client::new(keys.clone());
        client.add_relay(&relay_url).await?;
        client.connect().await;

        Ok(NostrDlc {
            keys,
            relay_url,
            client,
        })
    }

    pub fn start<S: Storage, O: Oracle>(
        &self,
        mut stop_signal: watch::Receiver<bool>,
        manager: Arc<DlcDevKitDlcManager<S, O>>,
    ) -> JoinHandle<Result<(), anyhow::Error>> {
        tracing::info!(
            pubkey = self.keys.public_key().to_string(),
            transport_public_key = self.public_key().to_string(),
            "Starting Nostr DLC listener."
        );
        let nostr_client = self.client.clone();
        let keys = self.keys.clone();
        tokio::spawn(async move {
            let since = Timestamp::now();
            let msg_subscription =
                nostr::messages::create_dlc_message_filter(since, keys.public_key());
            nostr_client.subscribe(msg_subscription, None).await?;
            tracing::info!(
                "Listening for messages on {}",
                keys.public_key().to_string()
            );
            let mut notifications = nostr_client.notifications();
            loop {
                tokio::select! {
                    _ = stop_signal.changed() => {
                        if *stop_signal.borrow() {
                            tracing::warn!("Stopping nostr dlc message subscription.");
                            nostr_client.disconnect().await;
                            break;
                        }
                    },
                    Ok(notification) = notifications.recv() => {
                        if let RelayPoolNotification::Event {
                            relay_url: _,
                            subscription_id: _,
                            event,
                        } = notification {
                            let (pubkey, message, event) = match handle_dlc_msg_event(
                                &event,
                                keys.secret_key(),
                            ) {
                                Ok(msg) => {
                                    tracing::info!(pubkey=msg.0.to_string(), "Received DLC nostr message.");
                                    (msg.0, msg.1, msg.2)
                                },
                                Err(_) => {
                                    tracing::error!("Could not parse event {}", event.id);
                                    continue;
                                }
                            };

                            match manager.on_dlc_message(&message, pubkey).await {
                                Ok(Some(msg)) => {
                                    let event = create_dlc_msg_event(
                                        event.pubkey,
                                        Some(event.id),
                                        msg,
                                        &keys,
                                    )
                                    .expect("no message");
                                    nostr_client
                                        .send_event(&event)
                                        .await
                                        .expect("Break out into functions.");
                                }
                                Ok(None) => (),
                                Err(_) => {
                                    // handle the error case and send
                                }
                            }
                        }
                    }
                }
            }
            Ok::<_, anyhow::Error>(())
        })
    }
}
