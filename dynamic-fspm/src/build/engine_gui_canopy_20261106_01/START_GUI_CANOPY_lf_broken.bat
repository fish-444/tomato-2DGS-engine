@echo off
rem GroIMP GUI canopy tool for the frozen Engine v2.2 (build/engine_gui_canopy_20261106_01). The frozen model files are not touched.
rem Working directory = gui\ (the model reads leaflet_outline_uv.csv from here; every run goes to gui\gui_runs\<time>\daily.csv).
chcp 65001 >nul
cd /d "%~dp0gui"
echo.
echo  토마토 캐노피 GUI (엔진 v2.2 시나리오 도구 - 검증 아님)
echo   A = SLA 수용 판 (잎면적 36 cm2)    R = SLA 비수용 판 (잎면적 24 cm2)
choice /c AR /n /m "  판을 고르세요 [A/R]: "
if errorlevel 2 (set "ARM=R") else (set "ARM=A")
set "JDK_JAVA_OPTIONS="
set "JAVA=C:\Program Files\Eclipse Adoptium\jre-21.0.12.101-hotspot\bin\java.exe"
if not exist "%JAVA%" set "JAVA=java"
set "CORE=C:\Program Files\GroIMP 2.2.1\core.jar"
echo.
echo  GroIMP 가 열리면: File - Open - 이 폴더의 TomatoCanopyGUI_sla%ARM%.gsz
echo  1) 오른쪽 편집기 맨 위 "설정" 숫자를 바꾸고 Ctrl+S (저장 = 다시 컴파일)
echo  2) 툴바 simulate 를 누르면 60일 실행, 차트가 매일 갱신됩니다
echo  3) 다시 돌리려면 설정을 저장(Ctrl+S)한 뒤 simulate
echo  결과: gui\gui_runs\^<시각^>\daily.csv    차트 창이 안 보이면 LIVE_CHARTS.bat 실행
echo  (메모리 1.5 GB 로 실행합니다. 여유 RAM 이 4 GB 미만이면 느리거나 멈출 수 있습니다.)
echo.
"%JAVA%" -Xmx1536m -Dfile.encoding=UTF-8 -Dtomato.gui=true -jar "%CORE%"
