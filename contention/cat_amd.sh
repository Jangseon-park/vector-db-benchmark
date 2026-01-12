#!/bin/bash
set -e

echo "[*] Mounting resctrl..."
sudo mount -t resctrl resctrl /sys/fs/resctrl || echo "resctrl already mounted."

cd /sys/fs/resctrl

echo "[*] Creating groups..."
sudo mkdir -p groupA
sudo mkdir -p groupB

echo "[*] Configuring groupA..."
sudo bash -c 'echo "L3:0=000f;2=000f;4=000f;6=000f;8=000f;10=000f;12=000f;14=000f;16=000f;18=000f;20=000f;22=000f;24=000f;26=000f;28=000f;30=000f" > groupA/schemata'
sudo bash -c 'echo 00000000,00000000,00000000,0000ffff > /sys/fs/resctrl/groupA/cpus'

echo "[*] Configuring groupB..."
sudo bash -c 'echo "L3:0=fff0;2=fff0;4=fff0;6=fff0;8=fff0;10=fff0;12=fff0;14=fff0;16=fff0;18=fff0;20=fff0;22=fff0;24=fff0;26=fff0;28=fff0;30=fff0" > groupB/schemata'
sudo bash -c 'echo 00000000,00000000,ffffffff,ffff0000 > /sys/fs/resctrl/groupB/cpus'

echo "[*] Done. Current resctrl groups:"
ls -l /sys/fs/resctrl

cat /sys/fs/resctrl/cpus
cat /sys/fs/resctrl/groupA/cpus
cat /sys/fs/resctrl/groupB/cpus
