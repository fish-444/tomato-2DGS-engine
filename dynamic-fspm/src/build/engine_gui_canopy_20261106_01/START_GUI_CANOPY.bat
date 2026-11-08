@echo off
rem GroIMP GUI canopy tool for the frozen Engine v2.2 (build/engine_gui_canopy_20261106_01). The frozen model files are not touched.
rem Working directory = gui\ ; every run goes to gui\gui_runs\<time>\daily.csv. Written by make_launchers.py with CRLF line endings.
chcp 65001 >nul
cd /d "%~dp0gui"
echo.
echo  토마토 캐노피 GUI (엔진 v2.2 시나리오 도구 - 검증 아님)
echo   A = SLA 수용 판 (잎면적 36 cm2)    R = SLA 비수용 판 (잎면적 24 cm2)
set "ARM=A"
choice /c AR /n /m "  판을 고르세요 [A/R]: "
if errorlevel 2 set "ARM=R"
set "JDK_JAVA_OPTIONS="
set "JAVA=C:\Program Files\Eclipse Adoptium\jre-21.0.12.101-hotspot\bin\java.exe"
if not exist "%JAVA%" set "JAVA=java"
set "CORE=C:\Program Files\GroIMP 2.2.1\core.jar"
set "PROJ=%CD%\TomatoCanopyGUI_sla%ARM%.gsz"
echo.
echo  선택: %ARM%   프로젝트: %PROJ%
echo  GroIMP 가 열리면 프로젝트가 자동으로 열립니다. 안 열리면 File - Open 으로 위 파일을 여세요.
echo  1) 오른쪽 편집기 맨 위 "설정" 숫자를 바꾸고 Ctrl+S (저장 = 다시 컴파일)
echo  2) 툴바 simulate 를 누르면 60일 실행, 차트가 매일 갱신됩니다
echo  결과: gui\gui_runs\^<시각^>\daily.csv    차트 창이 안 보이면 LIVE_CHARTS.bat 실행
echo  이 창은 GroIMP 를 닫을 때까지 열어 두세요.
echo.
"%JAVA%" -Xmx1536m -Dfile.encoding=UTF-8 -Dtomato.gui=true -jar "%CORE%" "%PROJ%"
echo.
echo  GroIMP 가 끝났습니다 (종료 코드 %ERRORLEVEL%). 오류가 보이면 이 창 내용을 복사해 주세요.
pause
