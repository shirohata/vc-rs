use anyhow::{bail, Context, Result};

pub(super) struct FeatureTensor {
    pub(super) data: Vec<f32>,
    pub(super) shape: Vec<i64>,
}

impl FeatureTensor {
    pub(super) fn repeat_frames(&mut self, factor: usize) -> Result<()> {
        if factor <= 1 {
            return Ok(());
        }
        if self.shape.len() != 3 {
            bail!("feature tensor must be rank-3 [1, frames, channels]");
        }
        let batch = usize::try_from(self.shape[0]).context("invalid feature batch")?;
        let frames = usize::try_from(self.shape[1]).context("invalid feature frames")?;
        let channels = usize::try_from(self.shape[2]).context("invalid feature channels")?;
        if batch != 1 {
            bail!("feature batch must be 1, got {batch}");
        }
        let mut repeated = Vec::with_capacity(self.data.len() * factor);
        for frame in 0..frames {
            let start = frame * channels;
            let end = start + channels;
            for _ in 0..factor {
                repeated.extend_from_slice(&self.data[start..end]);
            }
        }
        self.data = repeated;
        self.shape[1] = (frames * factor) as i64;
        Ok(())
    }

    pub(super) fn trim_front_frames(&mut self, frames_to_drop: usize) -> Result<()> {
        if frames_to_drop == 0 {
            return Ok(());
        }
        if self.shape.len() != 3 {
            bail!("feature tensor must be rank-3 [1, frames, channels]");
        }
        let batch = usize::try_from(self.shape[0]).context("invalid feature batch")?;
        let frames = usize::try_from(self.shape[1]).context("invalid feature frames")?;
        let channels = usize::try_from(self.shape[2]).context("invalid feature channels")?;
        if batch != 1 {
            bail!("feature batch must be 1, got {batch}");
        }
        if frames_to_drop >= frames {
            return Ok(());
        }
        let sample_offset = frames_to_drop * channels;
        self.data.drain(..sample_offset);
        self.shape[1] = (frames - frames_to_drop) as i64;
        Ok(())
    }
}
