from preprocessing.CICIDS_2017.pipeline_multi_csv import prepare_dataset_multi_csv
def preprocessing_CICIDS():
    train_files = [
        "datasets/CICIDS_2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "datasets/CICIDS_2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "datasets/CICIDS_2017/Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "datasets/CICIDS_2017/Monday-WorkingHours.pcap_ISCX.csv"
    ]

    test_files = [
        "datasets/CICIDS_2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "datasets/CICIDS_2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "datasets/CICIDS_2017/Tuesday-WorkingHours.pcap_ISCX.csv",
        "datasets/CICIDS_2017/Wednesday-workingHours.pcap_ISCX.csv"
    ]

    categorical = []         
    label_col = "Label"     
    clip_p = 0.99

    train_data, test_data, label_encoder, preprocessor = prepare_dataset_multi_csv(
        train_files,
        test_files,
        categorical,
        label_col,
        clip_p,
        "preprocessing/CICIDS_2017/saved_preprocessor_UNSW_NB15.pkl",
        "preprocessing/CICIDS_2017/label_encoder_UNSW_NB15.pkl"
    )
    return train_data, test_data, label_encoder, preprocessor
