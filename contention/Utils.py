import psutil
import os
import pandas as pd
import matplotlib.pyplot as plt

import invoke
import typer
    
def run(cmd, sudo=False, *args, **kwargs):
    kwargs.setdefault("echo", True)

    if not sudo:
        return invoke.run(cmd, *args, **kwargs)
    else:
        HOST_PASSWORD = "Pjs19571!"
        sudo_pass_responder = invoke.Responder(
            pattern=r"\[sudo\] password:.*", response=f"{HOST_PASSWORD}\n"
        )
        if HOST_PASSWORD == "unknown_host":
            print(
                "Please set the user password from env 'USER_PASSWORD'"
                "and call  again"
            )
            raise typer.Exit(1)

        kwargs.setdefault("pty", True)
        kwargs.setdefault("watchers", [sudo_pass_responder])
        return invoke.sudo(
            cmd,
            *args,
            **kwargs,
        )

class Utils:
    @staticmethod
    def run_proc(cmd_name):
        p = psutil.Process()
        os.system(cmd_name)
        print('Run child process %s (%s)...' % (cmd_name, os.getpid()))

    def parse_log_file(self, log_file="results.csv"):
        df = pd.read_csv(log_file)
        df['time_ns'] = df['time_ns'] - df['time_ns'].iloc[0]
        df['time_ms'] = df['time_ns'] / 1_000_000.0
        return df

    def plot_data_from_df(self, df, output_filename="side-channel.pdf"):
        if df is None or df.empty:
            print("DataFrame is empty or None. Cannot plot.")
            return

        plt.figure(figsize=(40, 4)) # Keep existing figsize
        plt.plot(df['time_ms'], df['latency_ns'], linestyle='-')
        
        plt.xlabel("Time (ms)")
        plt.ylabel("Latency (ns)")
        
        # plt.title("Latency over Time")
        plt.tight_layout()
        plt.grid(True)
        
        # Save the plot to a file
        try:
            plt.savefig(output_filename)
            print(f"Plot saved to {output_filename}")
        except Exception as e:
            print(f"Error saving plot: {e}")
        