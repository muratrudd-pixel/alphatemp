# Launchd Commands (macOS 26+)

## Install (one-time)
cp com.alphatemp.dashboard.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.alphatemp.dashboard.plist

## Stop
launchctl bootout gui/$(id -u)/com.alphatemp.dashboard

## Restart
launchctl kickstart -k gui/$(id -u)/com.alphatemp.dashboard

## Check status
launchctl print gui/$(id -u)/com.alphatemp.dashboard

## View logs
tail -f ~/Projects/alphatemp/alphatemp/logs/dashboard.log
