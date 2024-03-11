from dataset.dataloader_videoall import BaseFeeder
import torch
def test():
    feeder = BaseFeeder(None, gloss_dict=None, kernel_size=['K5', "P2", 'K5', "P2"])
    print(feeder)
    dataloader = torch.utils.data.DataLoader(

        dataset=feeder,
        batch_size=4,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=feeder.collate_fn,
    )
    for data in dataloader:
        pdb.set_trace()

if __name__ == '__main__':
    test()