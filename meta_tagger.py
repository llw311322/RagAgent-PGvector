"""元数据打标与数据清洗规则库

入库（vector_store.add_documents）与重打标（tools/tag_rebuild.py）共用这里的三类规则：

1. 清洗 clean_line / block_quality：
   - 行级：页码、签署栏（审核/校对/制图…）、分隔线、OCR 碎片行 → 剔除；
   - 块级：乱码比例过高（图示集扫描件）、有效字符过少的块 → 不入库。
2. 标准大类 classify_source：强制性通用规范 / 国家标准 / 行业标准 / 地方标准 / 参考资料。
3. 条文主题 classify_domain 与适用建筑类型 classify_buildings：
   按文件名 + 块的标题路径 + 正文开头联合判定，同一文件的不同条文块可以拿到不同标签
   （例如《全国民用建筑工程设计技术措施》里既有防火章也有节能章）。

所有字段写入块 metadata：
  standard_code  规范编号（GB55037-2022，等同 std_no）
  standard_name  规范名称（去掉编号与扩展名）
  clause_no      条文号（等同 article）
  chapter        所属节（等同 section）
  is_mandatory   是否强制性条文/全文强制规范
  valid          是否现行有效（征求意见稿、废止版本为 False）
  source_type    条文正文 / 条文说明 / 编制依据 / 图示
  std_class      标准大类（国标/行标/地标/参考资料/政策文件）
  std_no         规范编号（兼容旧字段）
  domain         条文专业主题
  buildings      适用建筑类型（列表）
  quality/garble 质检结果
"""
import re
from typing import List, Optional, Tuple

# ---------------------------------------------------------------- 清洗 ----

# 这些行对检索与阅读毫无价值（页眉页脚/签署栏/分隔线/编号行），直接剔除
_DIRTY_LINE = (
    re.compile(r"^\s*[-—_=·•\s.．~～]{5,}$"),                    # 分隔线
    re.compile(r"^\s*第?\s*\d{1,4}\s*页(\s*共\s*\d{1,4}\s*页)?\s*$"),  # 页码
    re.compile(r"^\s*\d{1,4}\s*$"),                              # 裸数字（OCR 页码）
    re.compile(r"^\s*(?:审核|校对|设计|制图|项目负责|审定|负责人|专业负责|工程负责|工程编号|图号)"),
    re.compile(r"^\s*图\s*集?\s*号\s*[:：]"),
    re.compile(r"^\s*[｜|│┃lI]{2,}\s*$"),                        # 竖线噪声
    re.compile(r"^\s*\(\s*\d{1,3}\s*\)\s*$"),                    # (12) 页脚编号
)

# 行内符号占比极高（OCR 碎片，如「1 N / -- ""」「〔〕〔〕」）
_SYMBOL_RUN = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9，。；：、（）《》%．,.;:()'\"\[\]-]")

_GARBLE_THRESHOLD = 0.28     # 乱码字符占比超过该值 → 噪声块
_MIN_VALID_CHARS = 20        # 有效字符（去脏行后）少于该值且无条文号 → 太碎，不入库


def clean_line(line: str) -> str:
    """行级清洗：脏行返回空串，其余原样返回。调用方据此丢弃该行但保留行号区间。"""
    s = line.strip()
    if not s:
        return line                      # 空行保留（正文段落分隔）
    if any(p.match(s) for p in _DIRTY_LINE):
        return ""
    return line


def garble_ratio(text: str) -> float:
    """乱码比例 0~1：中英文/数字/常用标点以外的字符占比。"""
    if not text:
        return 1.0
    good = 0
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff" or ch.isalnum():
            good += 1
        elif ch in "，。；：、（）《》%．,.;:()'\"\u3000 \n\t[]-—/|":
            good += 1
    return 1.0 - good / len(text)


def block_quality(text: str, article: str = "") -> Tuple[str, float]:
    """块级质检。返回 (quality, garble)：
    - "noise"  乱码比例过高（图示集扫描页），入库毫无价值；
    - "thin"   有效内容太少且无条文号（页眉碎片），检索只会添乱；
    - "ok"     正常块。
    """
    body = text.strip()
    g = garble_ratio(body)
    if g > _GARBLE_THRESHOLD:
        return "noise", g
    valid = sum(1 for ch in body if "\u4e00" <= ch <= "\u9fff" or ch.isalnum())
    if valid < _MIN_VALID_CHARS and not article:
        return "thin", g
    return "ok", g


# ------------------------------------------------------------ 标准大类 ----

# 图示集/图集/资料集/技术措施等参考读物——它们常带着 GB 号，必须先于国标判断
_REF_KEYS = ("图示", "图集", "资料集", "技术措施", "实施指南", "指南", "手册",
             "条文说明", "征求意见", "局部修订", "解读", "汇编", "强制性条文",
             "培训", "考试", "问答")
_STD_NO = re.compile(
    r"(GB\s*/?T?\s*Z?\s*5\d{4}|GB\s*/?T?\s*Z?\s*\d{2,5}|JGJ\s*\d{2,4}|CJJ\s*\d{2,4}"
    r"|CECS\s*\d{2,4}|JTG\s*[A-Z]?\s*\d{2,4}|DBJ?\s*/?\s*T?\s*\d{2,5})"
)


def classify_source(filename: str) -> Tuple[str, str]:
    """按文件名判定标准大类。返回 (std_class, std_no)。
    std_no 是文件里识别出的第一个标准号（如 GB 55037-2022），便于溯源。"""
    s = filename or ""
    m = _STD_NO.search(s)
    std_no = re.sub(r"\s+", "", m.group(1)) if m else ""
    if any(k in s for k in _REF_KEYS):
        return "参考资料", std_no
    # 中国工程规范体系里凡是《XX通用规范》都是全文强制的 GB 55xxx（如《建筑防火通用规范》
    # GB 55037）。很多 OCR 文件名丢了标准号，按名字兜底，否则会被判成"其他"输给专类规范。
    if "通用规范" in s or re.search(r"GB\s*/?T?\s*55\d{3}", s):
        return "国家标准-强制性通用规范", std_no
    if re.search(r"GB\s*/?T?\s*50\d{3}", s) or re.search(r"GB\s*\d{2,5}", s):
        return "国家标准", std_no
    if re.search(r"JGJ|CJJ|CECS|JTG|MH\s*\d|TB\s*\d|YD\s*\d|QB\s*\d|SB\s*\d", s):
        return "行业标准", std_no
    if re.search(r"DBJ?\s*/?\s*T?\s*\d", s):
        return "地方标准", std_no
    if re.search(r"住建部|住房和城乡建设厅|通知|管理办法|规定", s):
        return "政策文件", std_no
    return "其他", std_no


# ------------------------------------------------------------ 条文主题 ----

# 主题判定：文件名权重最高，标题路径次之，正文兜底（正文只看前 300 字，避免误判）
_DOMAIN_RULES = [
    ("防火", ("防火", "消防", "灭火", "喷水", "喷淋", "火灾", "防排烟", "排烟", "疏散",
             "避难", "耐火", "消防车道", "消防水源", "救援场地")),
    ("节能", ("节能", "绿色建筑", "绿建", "海绵", "可再生能源", "保温")),
    ("无障碍", ("无障碍", "适老化", "适老")),
    ("电气", ("电气", "电力", "照明", "智能建筑", "弱电", "防雷", "接地", "变电", "配电")),
    ("暖通燃气", ("暖通", "空调", "供暖", "采暖", "通风", "燃气", "锅炉", "热力")),
    ("给排水", ("给水", "排水", "给排水", "饮用水", "热水", "中水", "水景")),
    ("结构", ("结构", "混凝土", "钢结构", "抗震", "砌体", "地基", "基础", "木结构")),
    ("规划总图", ("规划", "总平面", "总图", "场地", "用地")),
    ("装饰装修", ("装修", "装饰", "幕墙", "屋面", "门窗")),
    ("声学光热", ("隔声", "声环境", "采光", "日照", "热工")),
]

_FY_STD_NO = re.compile(r"GB\s*5503[67]|GB\s*50(016|067|095|098|222|084|140|116)")


def classify_domain(source: str, heading: str = "", text: str = "") -> str:
    """条文所属专业主题（单标签，防火优先）。素材按权重：文件名 3 / 标题路径 2 / 正文前 300 字 1。"""
    materials = [
        (source or "", 3),
        (heading or "", 2),
        ((text or "")[:300], 1),
    ]
    best, best_score = "综合", 0
    for domain, keys in _DOMAIN_RULES:
        score = 0
        for mat, w in materials:
            hit = sum(1 for k in keys if k in mat)
            if hit:
                score += (min(hit, 3) + (2 if hit >= 2 else 0)) * w
        if score > best_score:
            best, best_score = domain, score
    # 文件名是防火/消防类标准号时直接定防火，正文关键词权不过它
    if best != "防火" and (_FY_STD_NO.search(source or "") or "防火" in (source or "")):
        best = "防火"
    return best


# ------------------------------------------------------------ 建筑类型 ----

_BUILDING_RULES = [
    ("住宅", ("住宅", "居住建筑", "保障房", "公寓", "宿舍")),
    ("酒店", ("酒店", "旅馆", "宾馆", "民宿", "招待所")),
    ("办公", ("办公", "写字楼")),
    ("商业", ("商业", "商场", "商店", "购物中心", "超市", "菜市场", "餐饮")),
    ("医疗", ("医院", "医疗", "疗养", "诊所", "疾控", "康复")),
    ("教育", ("学校", "中小学", "幼儿园", "托儿", "高校", "大学", "教学")),
    ("体育", ("体育", "场馆", "健身", "游泳池", "体育场", "体育馆")),
    ("文化", ("博物馆", "图书馆", "剧场", "影院", "展览", "文化馆", "美术馆")),
    ("汽车库", ("汽车库", "修车库", "停车场", "车库")),
    ("工业", ("工业", "厂房", "仓库", "物流", "生产", "多晶硅")),
    ("交通", ("车站", "机场", "航站楼", "码头", "港口", "铁路", "公路", "隧道")),
    ("轨道交通", ("地铁", "轨道交通", "城市轨道")),
    ("养老", ("养老", "老年人照料", "老年公寓")),
]
# 高层/地下是建筑形态，与用途并列单列，供「地库」「高层住宅」这类问题过滤
_FORM_RULES = (("地下", ("地下", "地库")), ("高层", ("高层",)))


def classify_buildings(source: str, heading: str = "", text: str = "") -> List[str]:
    """适用建筑类型（多标签，最多 4 个）。文件名命中权重 2，标题路径/正文 1。"""
    materials = [(source or "", 2), (heading or "", 1), ((text or "")[:300], 1)]
    tags: List[Tuple[str, int]] = []
    for tag, keys in _BUILDING_RULES:
        score = 0
        for mat, w in materials:
            hit = sum(1 for k in keys if k in mat)
            if hit:
                score += w * (1 if hit < 2 else 2)
        if score:
            tags.append((tag, score))
    tags.sort(key=lambda x: -x[1])
    out = [t for t, _ in tags[:3]]
    for form, keys in _FORM_RULES:
        if any(k in (source or "") or k in (heading or "") for k in keys):
            out.append(form)
    return out or ["通用"]


# ------------------------------------------------------- 名称/效力/类型 ----

_EXT_RE = re.compile(r"\.(?:md|markdown|txt|docx?|pdf)$", re.I)


def standard_name_of(filename: str) -> str:
    """从文件名还原规范名称：GB50016-2014(2018版)建筑设计防火规范.md → 建筑设计防火规范"""
    s = _EXT_RE.sub("", filename or "")
    # 顺序不能反：先删「(2018版)」这类版本括号，再删标准号。反过来会把左括号一起当分隔符吃掉，
    # 留下「版)建筑设计防火规范」这样的残渣。
    s = re.sub(r"[（(]\s*(?:19|20)\d{2}\s*版?\s*[)）]", " ", s)   # (2018版)
    s = _STD_NO.sub(" ", s)                                      # GB50016 / GB 50067 / JGJ 62
    s = re.sub(r"[-—–_]\s*(?:19|20)\d{2}\s*版?", " ", s)          # 尾部 -2014
    s = re.sub(r"^\s*[\d\-_.]+\s*", " ", s)                       # 序号前缀 01-05-05
    s = re.sub(r"\s+", " ", s).strip(" 《》-_·、")
    return s or (filename or "")


# 全文强制 / 含强制性条文的标志
_MANDATORY_KEYS = ("通用规范", "强制性条文", "强条")
# 非现行有效版本
_INVALID_KEYS = ("征求意见", "报批稿", "送审稿", "废止", "作废", "局部修订")
_DRAFT_IN_NAME = re.compile(r"征求意见|送审稿|报批稿|初稿")
_DRAFT_MARK = re.compile(r"^[（(【\[]?\s*(?:征求意见稿|送审稿|报批稿|讨论稿|初稿)\s*[)）】\]]?$")
# 明确宣告本文件失效的措辞。「原《XX》同时废止」指的是被替代的旧版，不算。
_ABOLISHED_IN_TEXT = re.compile(
    r"本(?:规范|标准|规程|通则|导则|图集|措施)[^\n。]{0,12}(?:已|被)(?:废止|作废|代替)"
    r"|(?:本(?:规范|标准|规程|通则|导则))\s*(?:已废止|已作废)"
)


def _has_draft_mark(head_text: str) -> bool:
    """正文里是否有「草案版本标识」。

    只认**独占一行**的标识（「（征求意见稿）」），不做全文关键词搜索。原因：
      1. 每本规范的「前言」都写「在广泛征求意见的基础上修订」，全文搜索会误判；
      2. 有的文件把被替代旧版的报批稿页眉一起 OCR 进来（如「GB50099-2008 …(报批稿)」），
         关键词搜索会把它当成草案。
    独占一行的写法是封面/扉页的固定格式，误判率最低。
    """
    for line in (head_text or "").splitlines():
        if line.strip() and _DRAFT_MARK.match(line.strip()):
            return True
    return False


def is_mandatory(std_class: str, source: str, heading: str = "", text: str = "") -> bool:
    """是否强制性：强制性通用规范（全文强制），或文件名/标题标注了「强制性条文」。

    不用「必须/严禁」这类措辞判断——规范里到处都有这两个词，按它打标会把大量普通条文
    标成强条，依据卡片上的「强制性条文」徽章就失去意义了。"""
    if std_class == "国家标准-强制性通用规范":
        return True
    return any(k in (source or "") or k in (heading or "") for k in _MANDATORY_KEYS)


def is_valid(source: str, head_text: str = "") -> bool:
    """是否现行有效：征求意见稿/报批稿/已废止版本为 False。"""
    return doc_status(source, head_text) == "valid"


def doc_status(source: str, head_text: str = "") -> str:
    """文档有效性：valid / draft（征求意见稿、送审稿、报批稿）/ abolished（已废止）。

    为什么要看正文而不只看文件名：
        库里有《建筑防火通用规范》两份——正式版「《建筑防火通用规范》GB 55037-2022.md」
        和草案「建筑防火通用规范.md」（文件名不带"征求意见"，正文首页才写着
        「（征求意见稿）」）。问「防火规范 5.2.2」时，草案文件名更短、与问题词面更贴合，
        会排在正式版前面，等于拿未生效的条文当标准答案。所以必须从正文首部判定。
    head_text 只传文档开头 800 字（分块时传全文开头、重打标时传首块），避免正文中途
    引用「某稿已被废止」之类表述造成误判。
    """
    head = (head_text or "")[:800]
    if _DRAFT_IN_NAME.search(source or "") or _has_draft_mark(head):
        return "draft"
    if "废止" in (source or "") or "作废" in (source or "") or _ABOLISHED_IN_TEXT.search(head):
        return "abolished"
    return "valid"


# 块的文本类型：条文正文 / 条文说明 / 编制依据 / 图示
def source_type_of(source: str, heading: str, text: str) -> str:
    """判定块的文本类型。规范类文档常在条文后附【编制依据】引用清单、末尾附「条文说明」，
    这两类内容带条文号、词面相似度极高，但都不是条文本身，检索时必须区分。"""
    h, t = heading or "", (text or "")[:200]
    if "条文说明" in h or "条文说明" in t:
        return "条文说明"
    if "编制依据" in h or "【编制依据】" in t:
        return "编制依据"
    if any(k in (source or "") for k in ("图示", "图集", "图册")) or any(
        k in h for k in ("图示", "图集")
    ):
        return "图示"
    return "条文正文"


# ------------------------------------------------------------ 打标入口 ----

def tag_chunk(meta: dict, text: str, status: str = "") -> dict:
    """给单个块补全结构化元数据（在原 metadata 上就地补充），返回该 metadata。

    status：文档级有效性（见 doc_status）。分块时由调用方按整篇判一次传入，避免逐块误判；
    不传则按本块文本自行判定（通用路径兜底）。
    """
    source = meta.get("source", "")
    std_class, std_no = classify_source(source)
    heading = meta.get("heading") or ""
    meta["std_class"] = std_class
    if std_no:
        meta["std_no"] = std_no
        meta["standard_code"] = std_no
    meta["standard_name"] = standard_name_of(source)
    # 条文号：结构化分块已写入 article，这里同步一份符合规范库命名习惯的字段
    if meta.get("article"):
        meta["clause_no"] = meta["article"]
    if meta.get("section"):
        meta["chapter"] = meta["section"]
    meta["is_mandatory"] = is_mandatory(std_class, source, heading, text)
    meta["valid"] = is_valid(source, text if not status else "")
    meta["source_type"] = source_type_of(source, heading, text)
    meta["domain"] = classify_domain(source, heading, text)
    meta["buildings"] = classify_buildings(source, heading, text)
    quality, garble = block_quality(text, meta.get("article", ""))
    meta["quality"] = quality
    meta["garble"] = round(garble, 3)

    # 草案/已废止：只标记「非现行」，不改变标准大类。
    # 为什么不降为参考资料：《建筑防火通用规范》的正式版文件是残缺的（75 块、仅 3 条），
    # 完整全文只存在于「征求意见稿」里。压到参考资料会让问「防火规范 5.2.2」时输给
    # 《汽车库…防火规范》，答案直接换错。所以保留权威性（通用规范仍高于专类规范），
    # 只标 valid=False 扣 0.10 分——正式版存在时自然排在前面，没有正式版时仍能顶上。
    st = status or doc_status(source, text)
    meta["doc_status"] = st
    if st in ("draft", "abolished"):
        meta["valid"] = False
        meta["is_mandatory"] = False
    return meta
