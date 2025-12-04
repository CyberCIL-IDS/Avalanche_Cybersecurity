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

    categorical = []          # se ne hai
    label_col = "Label"       # nel tuo dataset
    clip_p = 0.99

    train_data, test_data, label_encoder = prepare_dataset_multi_csv(
        train_files,
        test_files,
        categorical,
        label_col,
        clip_p,
        "saved_preprocessor.pkl",
        "saved_label_encoder.pkl"
    )
    return train_data, test_data, label_encoder
