
from data.sepsis_datamodule_cvae import GenericDataset, SepsisDataModule



if __name__ == "__main__":
    trace_attributes = [
    # "DiagnosticArtAstrup",
    {
        "name": "DiagnosticArtAstrup",
        "type": "categorical",
    }
    ]
    
    # dataset = GenericDataset(
    #     dataset_path='C:/Users/nikol/MT-repo/data/raw/Sepsis Cases - Event Log.xes',
    #     max_trace_length=100,
    #     num_activities=10,
    #     num_labels=2,
    #     trace_attributes=trace_attributes)
     
    datamodule = SepsisDataModule()
    datamodule.prepare_data()
    datamodule.setup()
    
    train_dataloader = datamodule.train_dataloader()
    for batch in train_dataloader:
        print(batch)
    test=1
        