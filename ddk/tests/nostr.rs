mod test_util;

#[cfg(feature = "nostr")]
mod nostr_test {
    use super::*;
    use bitcoin::{key::rand::Fill, Network};
    use chrono::{Local, TimeDelta};
    use ddk::oracle::memory::MemoryOracle;
    use ddk::storage::memory::MemoryStorage;
    use ddk::transport::nostr::NostrDlc;
    use ddk::DlcDevKit;
    use ddk::{builder::Builder, Transport};
    use dlc::{EnumerationPayout, Payout};
    use std::sync::Arc;

    type NostrDlcDevKit = DlcDevKit<NostrDlc, MemoryStorage, MemoryOracle>;

    async fn nostr_ddk(name: &str, oracle: Arc<MemoryOracle>) -> NostrDlcDevKit {
        let mut seed = [0u8; 32];
        seed.try_fill(&mut bitcoin::key::rand::thread_rng())
            .unwrap();
        let esplora_host = "http://127.0.0.1:30000".to_string();

        let transport = Arc::new(
            NostrDlc::new(&seed, "wss://nostr.dlcdevkit.com", Network::Regtest)
                .await
                .unwrap(),
        );
        let storage = Arc::new(MemoryStorage::new());

        let ddk: NostrDlcDevKit = Builder::new()
            .set_network(Network::Regtest)
            .set_seed_bytes(seed)
            .set_esplora_host(esplora_host)
            .set_name(name)
            .set_oracle(oracle)
            .set_transport(transport)
            .set_storage(storage)
            .finish()
            .await
            .unwrap();
        ddk
    }

    const EVENT_ID: &str = "nostr-event";

    #[test_log::test(tokio::test)]
    async fn nostr_contract() {
        let oracle = Arc::new(MemoryOracle::default());
        let alice = nostr_ddk("alice-nostr", oracle.clone()).await;
        let bob = nostr_ddk("bob-nostr", oracle.clone()).await;

        alice.start().unwrap();
        bob.start().unwrap();

        let alice_address = alice.wallet.new_external_address().await.unwrap().address;
        let bob_address = bob.wallet.new_external_address().await.unwrap().address;
        test_util::fund_addresses(&alice_address, &bob_address);

        let expiry = TimeDelta::seconds(15);
        let timestamp: u32 = Local::now()
            .checked_add_signed(expiry)
            .unwrap()
            .timestamp()
            .try_into()
            .unwrap();

        let announcement = oracle
            .oracle
            .create_enum_event(
                "nostr-event".to_string(),
                vec!["cat".to_string(), "ctv".to_string()],
                timestamp,
            )
            .await
            .unwrap();

        let contract_input = ddk_payouts::enumeration::create_contract_input(
            vec![
                EnumerationPayout {
                    outcome: "cat".to_string(),
                    payout: Payout {
                        offer: 100_000_000,
                        accept: 0,
                    },
                },
                EnumerationPayout {
                    outcome: "ctv".to_string(),
                    payout: Payout {
                        offer: 0,
                        accept: 100_000_000,
                    },
                },
            ],
            100_000_000,
            100_000_000,
            1,
            oracle.oracle.public_key().to_string(),
            EVENT_ID.to_string(),
        );
        let alice_pubkey = alice.transport.public_key();
        let _offer = bob
            .send_dlc_offer(&contract_input, alice_pubkey, vec![announcement])
            .await
            .unwrap();
    }
}
