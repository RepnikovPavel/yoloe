apt-get remove -y python3-blinker
cd ultralytics/
pip install -e ".[dev,export,solutions,logging,extra,typing]"
