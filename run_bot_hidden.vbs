' Starts run_bot_headless.bat without a console window (used by the OptionsBot scheduled task).
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
here = fso.GetParentFolderName(WScript.ScriptFullName)
sh.CurrentDirectory = here
sh.Run "cmd /c """ & here & "\run_bot_headless.bat""", 0, False
