@echo off
rem Open the frozen Engine v2.2 model (build/engine_v2_calib_20261104_01/work/Model_v2_2.rgg; v2.1: START_GUI_V2_1.bat, v2: START_GUI_V2_0.bat) in the GroIMP GUI. The v1 engine is untouched.
rem Working directory = work\gui_v2_2 (Model.rgg = copy of Model_v2_2.rgg, leaflet_outline_uv.csv next to it; runs write their csv files here).
chcp 65001 >nul
cd /d "%~dp0work"
if not exist gui_v2_2 mkdir gui_v2_2
copy /y Model_v2_2.rgg gui_v2_2\Model.rgg >nul
copy /y leaflet_outline_uv.csv gui_v2_2\leaflet_outline_uv.csv >nul
cd gui_v2_2
set "JDK_JAVA_OPTIONS="
set "JAVA=C:\Program Files\Eclipse Adoptium\jre-21.0.12.101-hotspot\bin\java.exe"
if not exist "%JAVA%" set "JAVA=java"
set "CORE=C:\Program Files\GroIMP 2.2.1\core.jar"
echo.
echo  Engine v2.2 (동결, 잎면적 36 cm²·복엽 길이 순위별·소엽 동일·키 12 cm) 모델을 GroIMP 로 엽니다. 원본 v1 은 그대로입니다.
echo  작업 폴더: %CD%
echo  GroIMP 가 열리면: File - Open - 이 폴더의 Model.rgg, 그다음 메서드 simulate 실행 (72일, 한 개체, 점광원)
echo  메모리 1 GB 이하로 실행합니다. 이 PC 는 여유 RAM 이 작아 긴 실행은 Kaggle 을 권합니다.
echo.
"%JAVA%" -Xmx1g -XX:ActiveProcessorCount=1 -Dfile.encoding=UTF-8 -jar "%CORE%"
