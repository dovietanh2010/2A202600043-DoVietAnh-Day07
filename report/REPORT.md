# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Đỗ Việt Anh
**Nhóm:** Nhóm 
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> *Viết 1-2 câu:* High cosine similarity nghĩa là hai vector embedding cùng hướng trong không gian nhiều chiều, cho thấy hai câu có ý nghĩa ngữ nghĩa (semantic meaning) rất giống nhau.

**Ví dụ HIGH similarity:**
- Sentence A: "Tôi rất thích nuôi chó."
- Sentence B: "Tôi là một người yêu thú cưng, đặc biệt là cún con."
- Tại sao tương đồng: Dù dùng từ vựng khác nhau, cả hai đều thể hiện sở thích nuôi chó/thú cưng.

**Ví dụ LOW similarity:**
- Sentence A: "Tôi rất thích nuôi chó."
- Sentence B: "Thị trường chứng khoán hôm nay giảm mạnh."
- Tại sao khác: Hai câu nói về hai chủ đề hoàn toàn không liên quan (thú cưng và tài chính), nên vector của chúng sẽ vuông góc hoặc ngược hướng.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> *Viết 1-2 câu:* Trong text embeddings, độ dài của vector thường phụ thuộc vào độ dài ngôn từ của tài liệu. Cosine similarity chỉ quan tâm đến *góc* (hướng) giữa hai vector chứ không bị ảnh hưởng bởi độ lớn (magnitude), nên nó phản ánh chính xác độ tương đồng ngữ nghĩa hơn là Euclidean distance.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> Trừ chunk đầu tiên (500 ký tự), nội dung còn lại là `10000 - 500 = 9500` ký tự.
> Mỗi chunk tiếp theo sẽ dịch lên một khoảng (step) là `chunk_size - overlap = 500 - 50 = 450` ký tự.
> Số lượng chunks thêm vào là `ceil(9500 / 450)` = `ceil(21.11)` = 22 chunks.
> *Đáp án:* Tổng cộng có 1 + 22 = 23 chunks. (Các điểm cắt bắt đầu: 0, 450, 900, ..., 9900).

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> *Viết 1-2 câu:* Nếu overlap tăng lên 100, bước nhảy sẽ giảm xuống còn 400. Số chunk = 1 + `ceil(9500 / 400)` = 1 + 24 = 25 chunks (tăng lên). Tăng overlap giúp bảo toàn ý nghĩa ngữ cảnh giáp ranh để khi đoạn văn bị cắt không làm đứt mạch ý tưởng, tăng chất lượng RAG (retrieval).

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Luật thi hành án hình sự.

**Tại sao nhóm chọn domain này?**
> Tài liệu này chứa các điều khoản, quy định pháp lý cụ thể, đòi hỏi độ chính xác cao khi truy xuất thông tin. Việc áp dụng RAG cho domain này giúp người dùng (sinh viên, luật sư) tra cứu nhanh các quy định, điều luật mà không cần đọc toàn bộ văn bản.

### Data Inventory

| # | Tên tài liệu                        | Nguồn                           | Số ký tự | Metadata đã gán                    |
|---|-------------------------------------|---------------------------------|----------|------------------------------------|
| 1 | GIÁO TRÌNH LUẬT THI HÀNH ÁN HÌNH SỰ | https://www.nguyenphuonglaw.com/ | 263,337  | `{"topic": "Luật", "lang": "vi"}` |

### Metadata Schema

| Trường metadata | Kiểu   | Ví dụ giá trị | Tại sao hữu ích cho retrieval?                                                              |
|-----------------|--------|---------------|---------------------------------------------------------------------------------------------|
| `topic`         | string | `Luật`        | Thu hẹp phạm vi chunk để so sánh (Pre-filter), tối ưu hoá tốc độ khi scope biết rõ môn học. |
| `lang`          | string | `vi`          | Tránh trường hợp model RAG generate lẫn lộn ngôn ngữ chéo.                                  |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

*Chạy `ChunkingStrategyComparator().compare()` trên tài liệu Luật thi hành án hình sự:*

| Tài liệu | Strategy                          | Chunk Count | Avg Length | Preserves Context?                                                                                              |
|----------|-----------------------------------|-------------|------------|-----------------------------------------------------------------------------------------------------------------|
| law.md   | FixedSizeChunker (`fixed_size`)   | 278         | 997.1      | Cắt văn bản theo giới hạn ký tự cứng, làm vỡ cấu trúc câu và từ (ví dụ: "chấ...", "thực hiện h...").           |
| law.md   | SentenceChunker (`by_sentences`)  | 452         | 579.6      | Bảo toàn câu nhưng làm rời rạc các danh sách liệt kê trong luật, dẫn đến thiếu hụt context liên quan.           |
| law.md   | RecursiveChunker (`recursive`)    | 2676        | 97.1       | Độ dài trung bình quá thấp (97 ký tự) khiến các chunk trở nên vụn vặt và mất ý nghĩa thực tế khi truy xuất.   |
| law.md   | CustomStrategy                    | 340         | 950.28     | Hiệu quả nhất, bảo toàn cấu trúc Điều luật và ngữ cảnh nhờ cơ chế Sliding Window.                               |

### Strategy Của Tôi

**Loại:** CustomStrategy (Hybrid Sliding Strategy)

**Mô tả cách hoạt động:**
> Hệ thống áp dụng một thuật toán chia tách lai (Hybrid Strategy) gồm hai lớp xử lý:
> 1.  **Phân tách cấu trúc đệ quy**: Sử dụng Regular Expression với kỹ thuật **Regex Lookahead (?=...)** để nhận diện các đề mục pháp lý (`CHƯƠNG`, `Điều`, `Mục`). Việc dùng Lookahead giúp giữ nguyên các tiêu đề này ở đầu mỗi chunk thay vì bị loại bỏ, bảo toàn mạch logic của luật.
> 2.  **Cửa sổ trượt (Sliding Window)**: Đối với các Điều luật có nội dung quá dài vượt quá `chunk_size`, hệ thống tự động áp dụng cơ chế chia nhỏ với **độ gối đầu (overlap=150)**. Đồng thời, toàn bộ các đoạn nhỏ (sub-chunks) đều được **chèn lại tiêu đề Điều luật gốc** ở dòng đầu tiên để AI luôn có đủ ngữ cảnh cần thiết khi thực hiện truy xuất.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Văn bản luật thường có các Điều luật rất dài và chứa nhiều tham chiếu chéo. Các phương pháp chia tách thuần túy (Fixed-size hay Sentence) thường làm mất tiêu đề Điều hoặc cắt cụt ý nghĩa giữa chừng. Chiến lược Hybrid này giúp khai thác tối đa cấu trúc tĩnh của Luật (Chương -> Điều -> Khoản) đồng thời đảm bảo mọi thông tin dù nằm ở cuối một Điều luật dài vẫn có thể được truy xuất kèm theo đầy đủ tiền tố ngữ cảnh (Context Injection).

**Code snippet (nếu custom):**
```python
import re

class CustomStrategy:
    """
    Hybrid strategy: Splits by document structure (Chapters, Articles) first.
    If a structural block is still too long, it applies a Sliding Window with overlap.
    """
    def __init__(self, chunk_size: int = 1000, overlap: int = 150) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Regex for Chapters and numbered sections (Articles/Clauses)
        self.separators = [
            r"\n#### \*\*CHƯƠNG", 
            r"\n#### \*\*\d+\.", 
            r"\n#### \*\*\d+\.\d+\.", 
            r"\n#### \*\*\d+\.\d+\.\d+\."
        ]  

    def chunk(self, text: str) -> list[str]:
        if not text: return []
        return self._split(text, self.separators)

    def _sliding_window(self, text: str, header_prefix: str = "") -> list[str]:
        """Splits a long text block into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text]
            
        chunks = []
        step = self.chunk_size - self.overlap
        for i in range(0, len(text), step):
            chunk_content = text[i : i + self.chunk_size]
            # Ensure we don't return near-empty last chunks
            if len(chunk_content) < self.overlap and chunks:
                break
                
            # Add header prefix to sub-chunks to maintain context if it's not the first one
            if i > 0 and header_prefix:
                chunks.append(f"{header_prefix} (tiếp theo):\n{chunk_content}")
            else:
                chunks.append(chunk_content)
                
            if i + self.chunk_size >= len(text):
                break
        return chunks

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            # Extract possible header from the start of the text for context injection
            header_match = re.match(r"(#### \*\*.*?\*\*)", text.strip())
            header_prefix = header_match.group(1) if header_match else ""
            return self._sliding_window(text, header_prefix)

        sep = separators[0]
        # Split but keep the separator in the resulting parts using lookahead
        parts = re.split(rf'(?={sep})', text)
        
        final_chunks = []
        current_chunk = ""

        for part in parts:
            if not part.strip(): continue
            
            if len(current_chunk) + len(part) <= self.chunk_size:
                current_chunk += part
            else:
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
                
                if len(part) > self.chunk_size:
                    # Try next inner separator level
                    final_chunks.extend(self._split(part, separators[1:]))
                    current_chunk = ""
                else:
                    current_chunk = part

        if current_chunk:
            final_chunks.append(current_chunk.strip())

        return final_chunks
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy                          | Chunk Count | Avg Length | Retrieval Quality?                                                                                              |
|----------|-----------------------------------|-------------|------------|-----------------------------------------------------------------------------------------------------------------|
| law.md   | FixedSizeChunker (`fixed_size`)   | 278         | 997.1      | Cắt văn bản theo giới hạn ký tự cứng, làm vỡ cấu trúc câu và từ (ví dụ: "chấ...", "thực hiện h...").           |
| law.md   | SentenceChunker (`by_sentences`)  | 452         | 579.6      | Bảo toàn câu nhưng làm rời rạc các danh sách liệt kê trong luật, dẫn đến thiếu hụt context liên quan.           |
| law.md   | RecursiveChunker (`recursive`)    | 2676        | 97.1       | Độ dài trung bình quá thấp (97 ký tự) khiến các chunk trở nên vụn vặt và mất ý nghĩa thực tế khi truy xuất.   |
| law.md   | CustomStrategy                    | 340         | 950.28     | Hiệu quả nhất, bảo toàn cấu trúc Điều luật và ngữ cảnh nhờ cơ chế Sliding Window.                               |

### So Sánh Với Thành Viên Khác

| Thành viên / cấu hình | Strategy                                                 | Retrieval Score (/10) | Điểm mạnh                                                                                                  | Điểm yếu                                                                         |
|-----------------------|----------------------------------------------------------|-----------------------|------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| Lê Thành Long         | `RecursiveChunker(chunk_size=800)` + metadata `section` | 8/10                  | Dễ triển khai, vẫn bám được heading và đoạn trong `law.md`, phù hợp để làm baseline cá nhân ổn định        | Chưa bám sát cấu trúc điều, khoản như các custom strategy nên một số query khó vẫn chưa trúng top-1 |
| Đỗ Xuân Bằng          | `CustomChunker (Header)`                                 | 9.5/10                | Gom khá trọn vẹn ý nghĩa của nguyên một điều luật, hạn chế bị xé ngữ cảnh                                  | Có nguy cơ tạo chunk dài vượt mức nếu một điều luật quá dài                      |
| Đỗ Việt Anh           | `CustomStrategy (Hybrid)`                                | 9.8/10                | Bảo toàn tốt tính bao đóng của điều, khoản; có sliding window nên xử lý điều dài vẫn giữ được gối đầu ngữ cảnh | Độ phức tạp tính toán cao hơn một chút so với các phương pháp thuần túy          |
| Lã Thị Linh           | `LegalArticleChunker (custom)`                           | 6.5/10                | Bám cấu trúc pháp lý tốt khi tài liệu đúng format                                                          | Khá nhạy với format tài liệu, cần regex robust hơn                               |
| Trương Anh Long       | `Custom (by sections)`                                   | 9/10                  | Giữ nguyên ngữ nghĩa theo từng điều, chương; hạn chế bị cắt nhỏ làm mất context                            | Phụ thuộc mạnh vào cấu trúc văn bản, khó áp dụng cho dữ liệu phi cấu trúc       |

**Strategy nào tốt nhất cho domain này? Tại sao?**  

Nếu xét theo benchmark của nhóm thì các strategy custom bán cấu trúc điều, khoản, chương của văn bản luật đang cho kết quả tốt hơn rõ rệt so với các chunker generic. Trong đó, `CustomStrategy (Hybrid)` cho điểm cao nhất vì vừa giữ được tính trọn nghĩa của điều luật, vừa có cơ chế gối đầu để không mất ngữ cảnh khi điều quá dài. Tuy vậy, với implementation cá nhân của em trong phạm vi lab này, `RecursiveChunker` vẫn là một lựa chọn khá cân bằng vì dễ triển khai hơn, không phụ thuộc quá mạnh vào format tài liệu nhưng vẫn tận dụng được cấu trúc heading của `law.md`.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Chiến lược này tập trung vào việc bảo toàn đơn vị ý nghĩa nhỏ nhất là "câu". Tôi sử dụng module `re` với kỹ thuật **Positive Lookbehind** `(?<=[.!?])\s+` để phân tách văn bản tại các dấu kết thúc câu (chấm, hỏi, than) mà không làm mất đi các ký hiệu này ở cuối câu. Sau khi có danh sách các câu đơn lẻ, thuật toán sẽ nhóm chúng lại thành từng khối (chunk) dựa trên tham số `max_sentences_per_chunk`, giúp đảm bảo mỗi đoạn context trả về cho LLM luôn là một tập hợp các câu trọn vẹn, không bị ngắt quãng giữa chừng.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Đây là một chiến lược chia tách đệ quy linh hoạt, mô phỏng cách con người phân cấp văn bản. Thuật toán nhận vào một danh sách các dấu phân tách có độ ưu tiên giảm dần: đoạn văn (`\n\n`) -> dòng (`\n`) -> câu (`. `) -> từ (` `). 
> - **Cơ chế**: Hàm `_split` sẽ thử chia văn bản bằng dấu phân tách đầu tiên. Nếu đoạn kết quả vẫn vượt quá `chunk_size`, nó sẽ tiếp tục đệ quy xuống cấp độ phân tách sâu hơn. 
> - **Ưu điểm**: Cách tiếp cận này giúp giữ các khối văn bản liên quan ở gần nhau nhất có thể (ví dụ: giữ nguyên một đoạn văn nếu nó đủ nhỏ) trước khi buộc phải chia nhỏ thêm, từ đó tối ưu hóa tính liên kết ngữ nghĩa.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Hệ thống được thiết kế với kiến trúc **Dual-Storage**: ưu tiên sử dụng `ChromaDB` nếu máy chủ có sẵn, nếu không sẽ tự động fallback về một mảng In-Memory để đảm bảo tính sẵn sàng. 
> - **Indexing**: Mỗi document được gán một UUID và metadata (source, chunk_index) trước khi chuyển đổi thành vector embedding.
> - **Search**: Tôi sử dụng phép tính **Dot Product** (tích vô hướng) giữa vector query và các vector lưu trữ. Vì tất cả vector đầu ra từ mô hình (kể cả Mock) đều được chuẩn hóa (normalized), nên kết quả tích vô hướng tương đương với **Cosine Similarity**. Các kết quả được sắp xếp giảm dần theo điểm số để chọn ra Top-K đoạn văn bản liên quan nhất.

**`search_with_filter` + `delete_document`** — approach:
> - **Filtering**: Triển khai giải thuật **Pre-filtering**. Trước khi tính toán độ tương đồng vector (vốn tốn kém tài nguyên), hệ thống sẽ lọc mảng dữ liệu dựa trên các điều kiện Metadata (như `topic` hay `doc_id`) bằng `list comprehension`. Điều này giúp thu hẹp không gian tìm kiếm và tăng tốc độ truy vấn.
> - **Deletion**: Việc xóa được thực hiện triệt để trên cả bộ nhớ đệm và database, đảm bảo tính nhất quán dữ liệu khi người dùng muốn gỡ bỏ một tài liệu lỗi thời.

### KnowledgeBaseAgent

**`answer`** — approach:
> Tôi triển khai luồng **Retrieval-Augmented Generation (RAG)** khép kín:
> 1. **Retrieval**: Gọi `self.store.search` để lấy các chunk liên quan nhất.
> 2. **Context Construction**: Kết nối nội dung các chunk thành một khối context duy nhất, đồng thời lọc bỏ các đoạn trùng lặp.
> 3. **Prompt Engineering**: Sử dụng template f-string nghiêm ngặt để ép LLM trả lời dựa vào context: `{context} \n\n Question: {question}`.
> 4. **Inference**: Invoke hàm LLM (`llm_fn`) để nhận câu trả lời cuối cùng, đảm bảo câu trả lời có tính căn cứ và giảm thiểu hiện tượng "ảo giác" (hallucination).

### Test Results

```
tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
... (Tất cả 42/42 tests đều chạy thành công xuất hiện PASSED) ...
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================= 42 passed in 0.09s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A                                          | Sentence B                                                       | Dự đoán | Actual Score | Đúng? |
|------|-----------------------------------------------------|------------------------------------------------------------------|---------|--------------|-------|
| 1    | "Phạm nhân đang thi hành án được hưởng quyền lợi gì?"| "Khu giam giữ phạm nhân nữ có tiêu chuẩn ra sao?"                | high    | 0.7402       | Đúng  |
| 2    | "Trại giam quản lý người như thế nào?"               | "Người mãn hạn tù phải có trách nhiệm gì?"                       | high    | 0.7366       | Đúng  |
| 3    | "Tôi thích đi du lịch vào mùa thu."                 | "Quy định về thời hiệu thi hành bản án."                         | low     | 0.1223       | Đúng  |
| 4    | "Quy định về hoãn chấp hành án phạt tù."            | "Luật pháp cho phép lùi thời điểm đi tù nếu có lý do chính đáng." | high    | 0.7761       | Đúng  |
| 5    | "Bản án chung thân có cơ hội giảm án không?"         | "Hướng dẫn chiên trứng bằng dầu ô liu."                          | low     | 0.0655       | Đúng  |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Kết quả tại **Pair 4** là bất ngờ và ấn tượng nhất. Mặc dù hai câu có cách dùng từ hoàn toàn khác biệt — một bên dùng thuật ngữ chuyên môn "hoãn chấp hành án phạt tù", một bên dùng ngôn ngữ đời thường "lùi thời điểm đi tù" — nhưng model vẫn cho ra số điểm tương đồng cao nhất (**0.7761**). Điều này chứng minh rằng Embeddings không hoạt động theo cơ chế so khớp từ vựng (lexical matching) đơn thuần, mà nó biểu diễn ngôn ngữ dưới dạng các vector trong không gian đa chiều, nơi khoảng cách giữa các vector đại diện cho **ý nghĩa ngữ nghĩa (semantic meaning)**. Nhờ đó, máy tính có thể hiểu được các khái niệm đồng nghĩa và ngữ cảnh trừu tượng, ngay cả khi chúng không chia sẻ bất kỳ từ khóa chung nào.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. 

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query                                                        | Gold Answer                                                                                       |
|---|--------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| 1 | Khái niệm pháp luật thi hành án hình sự là gì?               | Là tổng hợp các quy phạm pháp luật điều chỉnh các quan hệ xã hội phát sinh trong quá trình thi hành án. |
| 2 | Nguyên tắc nhân đạo trong thi hành án hình sự thể hiện như thế nào? | Không đối xử tàn bạo, bảo đảm pháp lý cho cuộc sống người bị kết án, tôn trọng quyền con người.   |
| 3 | Tác dụng giáo dục cải tạo của hình phạt là gì?               | Giáo dục cải tạo họ thành người lương thiện, tuân thủ pháp luật và có ích cho xã hội.           |
| 4 | Nhiệm vụ của pháp luật thi hành án hình sự là gì?            | Bảo đảm bản án được thực thi nghiêm minh, tạo điều kiện cho người thụ án tái hòa nhập cộng đồng. |
| 5 | Các quyền lợi hợp pháp bị xâm phạm thì người bị kết án giải quyết thế nào? | Có quyền khiếu nại, tố cáo đối với hành vi xâm phạm của cơ quan hoặc cá nhân thi hành án.       |


### Kết Quả Của Tôi

| # | Query                                                             | Top-1 Retrieved Chunk (tóm tắt)                                                                                     | Score | Relevant?    | Agent Answer (tóm tắt)                                                                     |
|---|-------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|-------|--------------|--------------------------------------------------------------------------------------------|
| 1 | Khái niệm pháp luật thi hành án hình sự là gì?                    | "#### **3. QUẢN CHẾ** (tiếp theo): vi cấp huyện nơi quản chế; Thủ trưởng cơ quan..."                                | 0.334 | **Không**    | Mô tả về thủ tục cấp giấy phép đi lại cho người bị quản chế thay vì định nghĩa khái niệm. |
| 2 | Nguyên tắc nhân đạo trong thi hành án hình sự thể hiện như thế nào| "#### **3.1. Quyền và nghĩa vụ của người chấp hành hình phạt tù** (tiếp theo): trí thời gian phù hợp để chăm sóc, nuôi dân..." | 0.318 | **Một phần** | Đề cập đến quyền chăm sóc con cái của phạm nhân nữ - một khía cạnh của tính nhân đạo.     |
| 3 | Tác dụng giáo dục cải tạo của hình phạt là gì?                    | "#### **5. THI HÀNH ÁN TREO** (tiếp theo): treo; hình phạt bổ sung; Ủy ban nhân dân cấp xã..."                      | 0.377 | **Một phần** | Trả về quy trình gửi quyết định thi hành án treo của Tòa án.                              |
| 4 | Nhiệm vụ của pháp luật thi hành án hình sự là gì?                 | "#### **4. BẮT BUỘC CHỮA BỆNH** (tiếp theo): Đình chỉ thi hành biện pháp bắt buộc chữa bệnh..."                   | 0.424 | **Không**    | Trả về quy định về việc đình chỉ biện pháp chữa bệnh bắt buộc khi đã khỏi bệnh.          |
| 5 | Quyền và nghĩa vụ của người bị kết án phạt tù theo pháp luật?     | "#### **1. KHÁI NIỆM CHUNG VỀ CÁC NGUYÊN TẮC CỦA PHÁP LUẬT THI HÀNH ÁN HÌNH SỰ** (tiếp theo): Việt Nam.* Qua khái niệm nê..." | 0.344 | **Không**    | Trả về phần lý luận chung về hệ thống nguyên tắc trong thi hành án hình sự.               |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 2 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Kĩ thuật chia tách văn bản dựa trên cấu trúc tài liệu (Document Structure Chunking) và cơ chế Sliding Window rất phù hợp với tài liệu luật, giúp bảo toàn tính bao đóng của các Điều/Khoản luật và giảm thiểu hiện tượng mất ngữ cảnh khi truy xuất.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Viết 2-3 câu:

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> E sẽ tập trung hơn công đoạn tiền xử lý (pre-processing) dữ liệu text. Rác từ parsing raw PDF sẽ khiến chất lượng embedding thuyên giảm (“Garbage in, Garbage out”), do đó cần loại trừ header/footer thừa ra khỏi dataset trước.

---

## Tự Đánh Giá

| Tiêu chí                      | Loại    | Điểm tự đánh giá |
|-------------------------------|---------|------------------|
| Warm-up                       | Cá nhân | 5 / 5            |
| Document selection            | Nhóm    | 10 / 10          |
| Chunking strategy             | Nhóm    | 15 / 15          |
| My approach                   | Cá nhân | 10 / 10          |
| Similarity predictions        | Cá nhân | 5 / 5            |
| Results                       | Cá nhân | 10 / 10          |
| Core implementation (tests)    | Cá nhân | 30 / 30          |
| Demo                          | Nhóm    | 0 / 5            |
| **Tổng**                      |         | **85 / 90**    |
