@echo off
REM =====================================================================
REM setup_and_run.bat — Tự tạo venv (nếu chưa có), cài đúng bộ thư viện
REM đã pin, rồi chạy onnx_to_tflite.py.
REM
REM Cách dùng: y hệt như gọi onnx_to_tflite.py, chỉ thêm .bat vào trước:
REM
REM   setup_and_run.bat --onnx weights\best_deploy.onnx --out weights\tflite ^
REM       --input-size 224 --mode all --calib-dir dataset\val --n-calib 200
REM
REM Lần đầu chạy sẽ tạo venv + cài thư viện (mất vài phút).
REM Các lần sau chạy lại sẽ tự động dùng venv có sẵn, không cài lại.
REM =====================================================================

setlocal

set VENV_DIR=venv-tflite
set REQ_FILE=requirements-tflite.txt
set PY=python

REM --- Kiểm tra python có tồn tại không ---
where %PY% >nul 2>nul
if errorlevel 1 (
    echo [LOI] Khong tim thay "python" trong PATH. Cai Python 3.10.11 truoc.
    exit /b 1
)

REM --- Kiểm tra requirements-tflite.txt có tồn tại cùng thư mục không ---
if not exist "%REQ_FILE%" (
    echo [LOI] Khong tim thay %REQ_FILE% trong thu muc hien tai.
    echo       Dat file nay cung thu muc voi setup_and_run.bat.
    exit /b 1
)

REM --- Tạo venv nếu chưa có ---
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [1/3] Chua co venv, dang tao "%VENV_DIR%" ...
    %PY% -m venv %VENV_DIR%
    if errorlevel 1 (
        echo [LOI] Tao venv that bai.
        exit /b 1
    )

    echo [2/3] Cai thu vien tu %REQ_FILE% ...
    "%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
    "%VENV_DIR%\Scripts\python.exe" -m pip install -r %REQ_FILE%
    if errorlevel 1 (
        echo [LOI] Cai thu vien that bai. Kiem tra log o tren.
        exit /b 1
    )
) else (
    echo [1/3] Venv "%VENV_DIR%" da co san, bo qua buoc tao + cai lai.
    echo       (Neu muon cai lai tu dau, xoa thu muc "%VENV_DIR%" roi chay lai script nay.)
)

REM --- Chạy script convert, forward toàn bộ tham số dòng lệnh ---
echo [3/3] Chay onnx_to_tflite.py ...
"%VENV_DIR%\Scripts\python.exe" onnx_to_tflite.py %*

endlocal
