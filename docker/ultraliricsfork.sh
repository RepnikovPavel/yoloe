apt-get remove -y python3-blinker
cd ultralyticsfork/
PIP_NO_CACHE_DIR=true PIP_ROOT_USER_ACTION=ignore pip install -e ".[dev,export,solutions,logging,extra,typing]"
