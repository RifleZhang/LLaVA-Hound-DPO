curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
# huggingface-cli lfs-enable-largefiles .

git config --global credential.helper cache
python -m pip install huggingface_hub