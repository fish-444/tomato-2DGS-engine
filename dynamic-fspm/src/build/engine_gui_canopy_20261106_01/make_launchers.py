"""Write START_GUI_CANOPY.bat and LIVE_CHARTS.bat with Windows CRLF line endings and UTF-8 (2026-11-07: the LF version did not open)."""
from pathlib import Path

H = Path(__file__).resolve().parent
START = [
    r'@echo off',
    r'rem GroIMP GUI canopy tool for the frozen Engine v2.2 (build/engine_gui_canopy_20261106_01). The frozen model files are not touched.',
    r'rem Working directory = gui\ ; every run goes to gui\gui_runs\<time>\daily.csv. Written by make_launchers.py with CRLF line endings.',
    r'chcp 65001 >nul',
    r'cd /d "%~dp0gui"',
    r'echo.',
    r'echo  토마토 캐노피 GUI (엔진 v2.2 시나리오 도구 - 검증 아님)',
    r'echo   A = SLA 수용 판 (잎면적 36 cm2)    R = SLA 비수용 판 (잎면적 24 cm2)',
    r'set "ARM=A"',
    r'choice /c AR /n /m "  판을 고르세요 [A/R]: "',
    r'if errorlevel 2 set "ARM=R"',
    r'set "JDK_JAVA_OPTIONS="',
    r'set "JAVA=C:\Program Files\Eclipse Adoptium\jre-21.0.12.101-hotspot\bin\java.exe"',
    r'if not exist "%JAVA%" set "JAVA=java"',
    r'set "CORE=C:\Program Files\GroIMP 2.2.1\core.jar"',
    r'set "PROJ=%CD%\TomatoCanopyGUI_sla%ARM%.gsz"',
    r'echo.',
    r'echo  선택: %ARM%   프로젝트: %PROJ%',
    r'echo  GroIMP 가 열리면 프로젝트가 자동으로 열립니다. 안 열리면 File - Open 으로 위 파일을 여세요.',
    r'echo  1) 오른쪽 편집기 맨 위 "설정" 숫자를 바꾸고 Ctrl+S (저장 = 다시 컴파일)',
    r'echo  2) 툴바 simulate 를 누르면 60일 실행, 차트가 매일 갱신됩니다',
    r'echo  결과: gui\gui_runs\^<시각^>\daily.csv    차트 창이 안 보이면 LIVE_CHARTS.bat 실행',
    r'echo  이 창은 GroIMP 를 닫을 때까지 열어 두세요.',
    r'echo.',
    r'"%JAVA%" -Xmx1536m -Dfile.encoding=UTF-8 -Dtomato.gui=true -jar "%CORE%" "%PROJ%"',
    r'echo.',
    r'echo  GroIMP 가 끝났습니다 (종료 코드 %ERRORLEVEL%). 오류가 보이면 이 창 내용을 복사해 주세요.',
    r'pause',
]
LIVE = [
    r'@echo off',
    r'rem Fallback chart window: reads the newest gui\gui_runs\<time>\daily.csv every 5 s (read-only).',
    r'chcp 65001 >nul',
    r'cd /d "%~dp0gui"',
    r'python live_charts.py',
    r'if errorlevel 1 pause',
]
for name, lines in (('START_GUI_CANOPY.bat', START), ('LIVE_CHARTS.bat', LIVE)):
    data = ('\r\n'.join(lines) + '\r\n').encode('utf-8')
    (H / name).write_bytes(data)
    bad = [c for c in data if c < 32 and c not in (10, 13)]
    lf_only = data.replace(b'\r\n', b'').count(b'\n')
    print(name, len(data), 'bytes; control chars:', len(bad), '; bare LF:', lf_only)
    for l in lines:
        if 'JAVA=' in l or 'PROJ=' in l or 'CORE=' in l: print('   ', l)
