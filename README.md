# Playing around with MNIST

* Create dataset artifact on WandB
    * Download MNIST data from pytorch github
    * Create WandB artifact
    * Store data in desired format on disk and add to artifact
        * train.pkl.gz
        * test.pkl.gz
    * Add visualization table to artifact
        * path
        * split
        * label
        * wandb.Image
* Train model
    * pl.LightningModule
    * WandB logging
        * train/val loss
        * train/val accuracy
        * predictions
        * model checkpoints
