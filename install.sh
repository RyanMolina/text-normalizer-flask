echo 'Installing virtual environment...'
python3 -m venv venv

echo 'Activating virtual environment...'
source venv/bin/activate

pip3 install -r requirements.txt
