
# Used method

pyinstaller AQiShell.spec
cd dist
./AQiShell

cd ~/Desktop
nano AQiShell.desktop

add:

[Desktop Entry]
Version=1.0
Name=AQiShell
Exec=/path/to/dist/AQiShell
Terminal=true
Type=Application
Icon=utilities-terminal
Comment=Launch AQiShell

chmod +x AQiShell.desktop
mv AQiShell.desktop ~/Desktop/

** click on icon **


# Other methods
pyinstaller --onefile --console AQiShell.py



