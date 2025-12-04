from preprocessing.pipeline_multi_csv import prepare_dataset_multi_csv

train_files = [
    "datasets_2\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "datasets_2\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "datasets_2\Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "datasets_2\Monday-WorkingHours.pcap_ISCX.csv"
]

test_files = [
    "datasets_2\Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "datasets_2\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "datasets_2\Tuesday-WorkingHours.pcap_ISCX.csv",
    "datasets_2\Wednesday-workingHours.pcap_ISCX.csv"
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
