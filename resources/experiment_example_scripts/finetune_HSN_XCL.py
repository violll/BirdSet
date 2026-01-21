import subprocess
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)


def main():
    subprocess.run(
        "python "
        + str(root / "birdset/train.py")
        + ' experiment="local/HSN/efficientnet_finetune_XCL"'
        + ' "module.network.model.local_checkpoint=' + str(root / "model.ckpt") + '"'
        + ' seed=2'
        + ' "trainer.devices=[0]"',
        shell=True,
    )


if __name__ == "__main__":
    main()
