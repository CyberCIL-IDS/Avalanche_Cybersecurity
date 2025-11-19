import numpy as np

def safe_ratio(a, b):
    return a / (b + 1)

def add_unsw_features(df):
    df["pkt_ratio"] = safe_ratio(df["spkts"], df["dpkts"])
    df["byte_ratio"] = safe_ratio(df["sbytes"], df["dbytes"])
    df["ttl_diff"] = df["sttl"] - df["dttl"]

    for col in ["sbytes", "dbytes", "sload", "dload", "dur"]:
        df[col + "_log"] = np.log1p(df[col])

    df["pkt_size_avg"] = safe_ratio(df["sbytes"] + df["dbytes"],
                                    df["spkts"] + df["dpkts"])

    return df
