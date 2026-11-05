//! Native dialogs need their own thread because they pump a nested window loop.
//! Track those threads at plugin scope so an open dialog cannot outlive the DLL.
use std::sync::Mutex;
use std::thread::{self, JoinHandle};

#[derive(Default)]
pub(crate) struct ModelPickers {
    threads: Mutex<Vec<JoinHandle<()>>>,
}

impl ModelPickers {
    pub(super) fn spawn(&self, pick: impl FnOnce() + Send + 'static) -> std::io::Result<()> {
        let mut threads = self.threads.lock().unwrap_or_else(|e| e.into_inner());
        let mut index = 0;
        while index < threads.len() {
            if threads[index].is_finished() {
                let _ = threads.swap_remove(index).join();
            } else {
                index += 1;
            }
        }
        threads.push(
            thread::Builder::new()
                .name("vc-vst3-picker".into())
                .spawn(pick)?,
        );
        Ok(())
    }
}

impl Drop for ModelPickers {
    fn drop(&mut self) {
        // The plugin/editor own this tracker; jobs capture only their settings
        // handles. Final teardown therefore joins on the owner, never on a
        // picker thread itself, and keeps plugin code loaded until dialogs exit.
        let threads = self.threads.get_mut().unwrap_or_else(|e| e.into_inner());
        for thread in threads.drain(..) {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{mpsc, Arc};
    use std::time::Duration;

    #[test]
    fn final_owner_waits_for_picker_even_after_editor_closes() {
        let plugin = Arc::new(ModelPickers::default());
        let editor = plugin.clone();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        editor
            .spawn(move || {
                started_tx.send(()).unwrap();
                release_rx.recv_timeout(Duration::from_secs(2)).unwrap();
            })
            .unwrap();
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        drop(editor);

        let (dropping_tx, dropping_rx) = mpsc::channel();
        let (dropped_tx, dropped_rx) = mpsc::channel();
        let owner = thread::spawn(move || {
            dropping_tx.send(()).unwrap();
            drop(plugin);
            dropped_tx.send(()).unwrap();
        });
        dropping_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(matches!(
            dropped_rx.recv_timeout(Duration::from_millis(50)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        release_tx.send(()).unwrap();
        dropped_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        owner.join().unwrap();
    }
}
