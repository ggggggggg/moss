import pytest
import moss

noise_path = r"C:\Users\oneilg\Desktop\python\src\mass\tests\ljh_files\20230626\0000\20230626_run0000_chan4102.ljh";
pulse_path = r"C:\Users\oneilg\Desktop\python\src\mass\tests\ljh_files\20230626\0001\20230626_run0001_chan4102.ljh";

def test_to_df():
    ljh_noise = moss.LJHFile(noise_path)
    df_noise, header_df_noise = ljh_noise.to_polars()
    ljh = moss.LJHFile(pulse_path)
    df, header_df = ljh.to_polars()
