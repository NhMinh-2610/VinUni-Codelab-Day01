# Student Guide: Nền tảng LLM API (Step-by-Step Walkthrough)

> [!NOTE]
> Tài liệu này được thiết kế để hỗ trợ bạn hoàn thành bài Lab 01 một cách nhanh chóng nhất. Hãy sử dụng nó làm cẩm nang khi triển khai các hàm trong `template.py`.

---

## 🛠️ 1. Khởi tạo Môi trường & Thiết lập API Keys

### 🔹 1.1 Khởi tạo môi trường ảo (venv) & Cài đặt thư viện
Để tránh xung đột thư viện giữa các dự án, bạn nên sử dụng môi trường ảo (`.venv`):

1. **Khởi tạo Virtual Environment (`.venv`):**
   * **Windows/macOS/Linux**:
     ```bash
     python -m venv .venv
     ```

2. **Kích hoạt môi trường ảo (Activate):**
   * **Windows (PowerShell)**:
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   * **Windows (CMD)**:
     ```cmd
     .venv\Scripts\activate.bat
     ```
   * **macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```

3. **Cài đặt thư viện từ [requirements.txt](requirements.txt):**
   ```bash
   pip install -r requirements.txt
   ```

---

### 🔹 1.2 Hướng dẫn lấy API Keys

* **Google Gemini API Key (GEMINI_API_KEY):**
  1. Truy cập trang web [Google AI Studio](https://aistudio.google.com/).
  2. Đăng nhập bằng tài khoản Google của bạn.
  3. Bấm vào nút **"Get API key"** (ở menu bên trái hoặc góc trên).
  4. Chọn **"Create API key"** và tạo khóa cho dự án mới hoặc dự án hiện có.
  5. Dùng nút sao chép để lấy toàn bộ API key đã tạo. Giữ nguyên key được cấp, không sửa tiền tố theo một chuỗi mẫu.

* **OpenAI API Key (OPENAI_API_KEY):**
  1. Truy cập [OpenAI API Keys Platform](https://platform.openai.com/api-keys).
  2. Đăng nhập/Đăng ký tài khoản OpenAI của bạn.
  3. Bấm vào nút **"Create new secret key"**.
  4. Đặt tên gợi nhớ cho key (ví dụ: `VinUni_Lab1`) và chọn quyền hạn (mặc định là All).
  5. Sao chép API key đã tạo (nó sẽ có dạng bắt đầu bằng `sk-proj-...`).
  > [!WARNING]
  > Key OpenAI chỉ hiển thị duy nhất **một lần** ngay khi tạo. Hãy lưu trữ key an toàn.

---

### 🔹 1.3 Thiết lập Biến môi trường trực tiếp (Environment Variables)

**Dùng chatbot OpenAI thay cho Gemini (Windows CMD):**

```cmd
.venv\Scripts\python.exe starter-code\template.py --provider openai
```

Tạo khóa OpenAI riêng tại [OpenAI API Keys](https://platform.openai.com/api-keys). Khi chương trình hiện `Paste OpenAI API key (input hidden):`, dán khóa vào terminal rồi nhấn Enter. Nội dung nhập được ẩn; không gửi khóa vào chat. Nếu đã đặt `OPENAI_API_KEY` hợp lệ trong terminal, chương trình sẽ dùng biến đó.

Chế độ này mặc định dùng `gpt-5.4-mini` và chỉ cần khóa OpenAI. Có thể chọn model khác mà project OpenAI của bạn có quyền dùng:

```cmd
.venv\Scripts\python.exe starter-code\template.py --provider openai --model gpt-5.4-mini
```

Bot hiển thị câu trả lời theo luồng và giữ ba lượt hội thoại hoàn tất gần nhất. Nhập `quit` hoặc `exit` để thoát. Chế độ này dùng [Responses API của OpenAI](https://developers.openai.com/api/docs/guides/streaming-responses); cần quyền truy cập API và quota/billing tương ứng. Nếu SDK cũ báo không có `responses`, chạy `python -m pip install -r requirements.txt` trong môi trường ảo.

**Cách đơn giản khi chỉ dùng Gemini:**

```powershell
python starter-code/template.py
```

Nếu chưa có key hoặc giá trị biến môi trường sai định dạng, chương trình sẽ hỏi `Paste Gemini API key (input hidden):`. Dán key thật tại lời nhắc này rồi nhấn Enter; nội dung nhập sẽ được ẩn trong terminal. Key được giữ trong phiên chạy, không được ghi vào file.

Nếu Google trả lỗi `API_KEY_INVALID`, chương trình sẽ cho nhập key thay thế. Tạo key mới trong Google AI Studio, sao chép đầy đủ rồi dán vào lời nhắc. Sau đó nhập lại câu hỏi để gửi bằng key mới.

Mặc định chương trình chỉ chạy chatbot Gemini. Để chạy thêm phần so sánh trực tiếp với OpenAI (tiêu thụ quota API), đặt `OPENAI_API_KEY` bằng khóa OpenAI riêng rồi chạy:

```powershell
python starter-code/template.py --compare
```

Khóa Gemini không dùng được với OpenAI. Nếu chỉ dùng Gemini và trước đó đã gán nhầm khóa vào `OPENAI_API_KEY`, có thể xóa biến đó trong phiên PowerShell hiện tại:

```powershell
Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
```

Nếu dùng **CMD** (dòng nhắc `C:\...>`), hãy dùng `set`, vì `Remove-Item` chỉ có trong PowerShell (dòng nhắc `PS C:\...>`). Để xóa khóa Gemini trong phiên CMD và nhập lại qua lời nhắc ẩn:

```cmd
set "GEMINI_API_KEY="
set "GOOGLE_API_KEY="
python starter-code/template.py
```

Bạn có thể thiết lập biến môi trường trực tiếp từ terminal (áp dụng cho phiên làm việc hiện tại) bằng các lệnh sau:

* **Windows (PowerShell):**
  ```powershell
  $env:GEMINI_API_KEY = (Read-Host "Dan key Gemini that vao day roi nhan Enter").Trim()
  ```
  Giữ nguyên dòng lệnh trên. Dán key thật sau khi terminal hiện lời nhắc, không thay nội dung trong `Read-Host` bằng key. Phải có đủ `()` sau `.Trim()`; viết `.Trim` sẽ lưu mô tả hàm vào biến môi trường thay vì key.

  Nếu cần chạy so sánh GPT-4o, GPT-4o-mini và Gemini, đặt thêm `OPENAI_API_KEY` bằng key OpenAI thật. Một key OpenAI dùng cho cả hai model GPT trong bài.

* **Windows (CMD):**
  ```cmd
  set OPENAI_API_KEY=sk-proj-your-openai-key-here
  set GEMINI_API_KEY=AIzaSy-your-gemini-key-here
  ```

* **macOS/Linux (Terminal):**
  ```bash
  export OPENAI_API_KEY="sk-proj-your-openai-key-here"
  export GEMINI_API_KEY="AIzaSy-your-gemini-key-here"
  ```

Các chuỗi chứa `your-...-key-here` trong ví dụ CMD/Bash là giá trị mẫu, cần thay bằng key thật.

> [!TIP]
> Chương trình đọc biến môi trường của terminal hoặc hỏi khóa qua lời nhắc ẩn; hiện không tự đọc file `.env`. Khóa nhập tại lời nhắc chỉ tồn tại trong tiến trình Python và không được ghi vào mã nguồn.

---

## 🚀 2. Lộ trình triển khai (Recommended Flow)

Để hoàn thành bài lab một cách khoa học, hãy đi theo trình tự sau:

    A[Task 1: call_openai] --> B[Task 2: call_gemini]
    B --> C[Task 3: call_anthropic]
    C --> D[Task 4: compare_models]
    D --> E[Task 5: streaming_chatbot]
    E --> F[Bonus A: retry_with_backoff]
    F --> G[Bonus B/C: batch & table]


---

## 🛠️ 3. Hướng dẫn chi tiết từng nhiệm vụ

### 🔹 Task 1: `call_openai` (OpenAI API)
* **SDK cần dùng:** `from openai import OpenAI`
* **Cách khởi tạo:** `client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))`
* **Hàm gọi:** `client.chat.completions.create(...)`
* **Tham số:**
  * `model`: chuỗi tên model (`gpt-4o` hoặc `gpt-4o-mini`).
  * `messages`: dạng danh sách hội thoại `[{"role": "user", "content": prompt}]`.
  * `temperature`, `top_p`, `max_tokens`.
* **Đo thời gian (Latency):** Dùng thư viện `time`:
  ```python
  start = time.time()
  # call API
  latency = time.time() - start
  ```
* **Lấy Token Usage:**
  ```python
  input_tokens = response.usage.prompt_tokens
  output_tokens = response.usage.completion_tokens
  ```

---

### 🔹 Task 2: `call_gemini` (Google Gemini 2.5) — Trọng tâm chính!
Chúng ta sẽ sử dụng SDK mới nhất của Google (`google-genai`):
* **SDK cần dùng:** `from google import genai` và `from google.genai import types`
* **Cách khởi tạo:** `client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))`
* **Hàm gọi:** `client.models.generate_content(...)`
* **Cấu hình tham số (Configuration):**
  ```python
  config = types.GenerateContentConfig(
      temperature=temperature,
      top_p=top_p,
      max_output_tokens=max_tokens
  )
  response = client.models.generate_content(
      model=model,
      contents=prompt,
      config=config
  )
  ```
* **Lấy Token Usage:**
  ```python
  input_tokens = response.usage_metadata.prompt_token_count
  output_tokens = response.usage_metadata.candidates_token_count
  ```

---

### 🔹 Task 5: `streaming_chatbot` (Gemini Streaming + History)
Streaming giúp tạo trải nghiệm người dùng mượt mà giống như ChatGPT (chữ chạy đến đâu hiển thị đến đó).
* **Streaming logic:**
  ```python
  response_stream = client.models.generate_content_stream(
      model="gemini-2.5-flash",
      contents=formatted_history
  )
  for chunk in response_stream:
      print(chunk.text, end="", flush=True)
  ```
* **Lịch sử hội thoại (History Management):**
  Giới hạn lịch sử ở 3 lượt hội thoại gần nhất (tương đương 6 tin nhắn).
  ```python
  # Cắt lịch sử nếu quá dài
  history = history[-6:]
  ```

---

## ✅ 4. Kiểm tra và nộp bài

Bản hoàn chỉnh đáp ứng năm task và ba bonus trong đề gốc:

- [x] Task 1: OpenAI Chat Completions, tham số tùy chọn, latency và token usage.
- [x] Task 2: Google GenAI, Gemini 2.5 Flash, cấu hình sampling và usage metadata.
- [x] Task 3: Anthropic Claude, ghép các khối văn bản và lấy token usage.
- [x] Task 4: So sánh GPT-4o, GPT-4o-mini và Gemini 2.5 Flash; tính chi phí theo bảng giá cố định của bài lab.
- [x] Task 5: Chatbot Gemini streaming, lưu ba lượt hội thoại gần nhất, thoát bằng `quit`/`exit`.
- [x] Bonus A: Retry với exponential backoff và giới hạn số lần thử.
- [x] Bonus B: Batch theo thứ tự prompt; tiếp tục sau prompt bị lỗi.
- [x] Bonus C: Bảng Markdown, response tối đa 50 ký tự, xử lý dấu phân cách và xuống dòng.

Các tham số công khai của tám hàm theo đề bài được giữ nguyên. `solution/solution.py` là bản nộp theo hướng dẫn gốc; `solution-code/solution.py` giữ bản tương thích với đường dẫn chấm bài trước đây. Hai file này phải giống hệt `starter-code/template.py`. Sau khi sửa mã, đồng bộ và kiểm tra:

```cmd
python scripts\sync_solutions.py
python scripts\sync_solutions.py --check
python -m pytest tests -v
```

Bộ test mặc định chạy trên `solution/solution.py`. Để kiểm tra bản khác, đặt `LAB_SOLUTION_PATH` trong CMD:

```cmd
set "LAB_SOLUTION_PATH=starter-code/template.py"
python -m pytest tests -v
set "LAB_SOLUTION_PATH="
```

Test dùng API giả lập nên không cần khóa thật, không gọi mạng và không tiêu thụ quota. GitHub Actions chạy toàn bộ test trên Python 3.11 và 3.13 cho cả ba đường dẫn. Tên model và bảng giá trong phần so sánh được giữ theo đề để chấm bài; giá trị `cost` là phép tính theo bảng của bài lab, không phải báo cáo thanh toán hiện tại. Nếu cả phép so sánh bị lỗi, batch trả hàng lỗi với metric bằng 0 để biểu thị dữ liệu không có; các request đã hoàn tất trước lỗi vẫn có thể bị nhà cung cấp tính phí.

## 🆘 5. Giải quyết các lỗi thường gặp (Troubleshooting)

### 🔴 Lỗi `ModuleNotFoundError: No module named 'google'`
* **Nguyên nhân:** Chưa cài đặt thư viện Google GenAI SDK.
* **Cách khắc phục:** Chạy `pip install google-genai` trong terminal của bạn.

### 🔴 Lỗi `AuthenticationError` hoặc `API Key Not Found`
* **Nguyên nhân:** API key chưa được thiết lập vào biến môi trường.
* **Cách khắc phục:**
  * **MacOS/Linux:** `export GEMINI_API_KEY="AIzaSy..."`
  * **Windows (PowerShell):** `$env:GEMINI_API_KEY="AIzaSy..."`
  * **Windows (CMD):** `set GEMINI_API_KEY=AIzaSy...`

### 🔴 Lỗi `RateLimitError`
* **Giới hạn ngắn hạn (RPM/TPM):** Bạn gửi quá nhiều request hoặc token trong một phút. Chờ theo thời gian retry mà API trả về; có thể dùng **Task Bonus A (`retry_with_backoff`)** với số lần thử hữu hạn.
* **Hết quota ngày (`429 RESOURCE_EXHAUSTED`, `RequestsPerDay` hoặc `requests per day`):** Chờ đến thời gian retry được báo, hoặc kiểm tra quota và billing của project tại [Google AI Studio](https://ai.dev/rate-limit). Thử lại sau vài giây không khôi phục quota ngày. Chatbot sẽ giải thích lỗi và kết thúc phiên để tránh gửi thêm request.
* Quota Gemini áp dụng **theo project**, không theo API key. Tạo key mới trong cùng project không đặt lại quota. Hạn mức thay đổi theo model và tier; xem [tài liệu rate limits của Google](https://ai.google.dev/gemini-api/docs/rate-limits).
* Banner của chatbot hiển thị giá trị `GEMINI_MODEL` thực tế trong `starter-code/template.py`; tên trong tài liệu bài lab có thể khác nếu bạn đã đổi model.

### 🔴 Lỗi `503 UNAVAILABLE` / `This model is currently experiencing high demand`
* Google đang thiếu năng lực phục vụ tạm thời cho model. Không cần đổi API key chỉ vì lỗi này; xem [hướng dẫn lỗi Gemini của Google](https://ai.google.dev/gemini-api/docs/generate-content/api-errors).
* Chatbot thử lại tối đa hai lần, sau 2 và 4 giây, khi chưa in nội dung trả lời. Nếu vẫn lỗi, chờ thêm rồi nhập lại câu hỏi tại `You:`.
* Nếu stream đã in một phần câu trả lời rồi bị lỗi, chương trình không tự gửi lại để tránh phát lại nội dung. Lượt bị lỗi không được thêm vào lịch sử hội thoại.

---
> **💡 Vibe Coding Tip:** Hãy tận dụng Cursor hoặc Copilot bằng cách cung cấp cho nó ngữ cảnh của file `student_guide.md` và `template.py` để viết code nháp nhanh chóng, sau đó kiểm tra và hoàn thiện bằng tay!
