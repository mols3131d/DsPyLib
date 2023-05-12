import numpy as np
import pandas as pd


def Light_Num_Col(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    - 소개
      pd.df의 숫자 컬럼들의 데이터 타입을 작게 변경하는 함수.
    - 목적
      램 사용량 감소
    """

    numerics = [
        "uint16",
        "uint32",
        "uint64",
        "int16",
        "int32",
        "int64",
        "float16",
        "float32",
        "float64",
    ]
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int" or str(col_type)[:4] == "uint":
                # uint
                if c_min >= 0:
                    if c_min >= np.iinfo(np.uint8).min and c_max <= np.iinfo(np.uint8).max:
                        df[col] = df[col].astype(np.uint8)
                    elif c_min >= np.iinfo(np.uint16).min and c_max <= np.iinfo(np.uint16).max:
                        df[col] = df[col].astype(np.uint16)
                    elif c_min >= np.iinfo(np.uint32).min and c_max <= np.iinfo(np.uint32).max:
                        df[col] = df[col].astype(np.uint32)
                    elif c_min >= np.iinfo(np.uint64).min and c_max <= np.iinfo(np.uint64).max:
                        df[col] = df[col].astype(np.uint64)

                # int
                else:
                    if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min >= np.iinfo(np.int64).min and c_max <= np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                    else:
                        df[col] = df[col].astype(np.int64)
            # float
            else:
                if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )

    return df
