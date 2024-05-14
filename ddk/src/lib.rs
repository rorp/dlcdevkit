#![allow(dead_code)]
#![allow(unused_imports)]
mod chain;
mod error;
mod io;
mod oracle;
mod signer;
mod wallet;

pub mod nostr_manager;
pub mod p2p;
pub use bdk;
pub use bitcoin::Network;
pub use dlc_manager;
pub use dlc_messages;
pub use dlc_sled_storage_provider::SledStorageProvider;
pub use io::get_dlc_dev_kit_dir;

pub const RELAY_HOST: &str = "ws://localhost:8081";
pub const ORACLE_HOST: &str = "http://localhost:8080";