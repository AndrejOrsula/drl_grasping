#!/usr/bin/env bash

echo "Setting up loopback network interface for multicasting..."

sudo apt-get update &&
sudo apt-get install -y net-tools

sudo route add -net 224.0.0.0 netmask 240.0.0.0 dev lo
sudo ifconfig lo multicast

sudo touch /etc/rc.local
sudo chmod 0775 /etc/rc.local
echo -e "#!/usr/bin/env bash\n\nifconfig lo multicast" | sudo tee -a /etc/rc.local
