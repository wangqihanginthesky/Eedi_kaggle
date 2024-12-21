def add_subject(df):
    """
    Add subject-related fields to the DataFrame.

    TODO: Replace dummy values with actual logic to populate subject fields.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with subject fields added.
    """
    df["FirstSubjectId"] = "dummy_id_1"
    df["FirstSubjectName"] = "dummy_name_1"
    df["SecondSubjectId"] = "dummy_id_2"
    df["SecondSubjectName"] = "dummy_name_2"
    df["ThirdSubjectId"] = "dummy_id_3"
    df["ThirdSubjectName"] = "dummy_name_3"
    return df
