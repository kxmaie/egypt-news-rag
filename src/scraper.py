import hashlib
import html
import json
import logging
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger("egypt_news_scraper")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

CAIRO_TZ = ZoneInfo("Africa/Cairo")

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ar-EG,ar;q=0.9,en-US;q=0.8,en;q=0.7",
}

DOMAIN_NAMES_AR = {
    "youm7.com": "اليوم السابع",
    "masrawy.com": "مصراوي",
    "almasryalyoum.com": "المصري اليوم",
    "shorouknews.com": "الشروق",
    "cairo24.com": "القاهرة 24",
    "ahram.org.eg": "الأهرام",
    "akhbarelyom.com": "أخبار اليوم",
    "elwatannews.com": "الوطن",
    "dostor.org": "الدستور",
    "almalnews.com": "المال",
}


class _DOMExtractor(HTMLParser):
    """مستخرج DOM خفيف ومعتمد على المكتبة القياسية لاستخراج الروابط ووسوم الميتا والفقرات النصية."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: List[Dict[str, str]] = []
        self.head_links: List[Dict[str, str]] = []
        self.metas: List[Dict[str, str]] = []
        self.json_ld_blocks: List[str] = []
        self.h1_texts: List[str] = []
        self.title_texts: List[str] = []
        self.paragraphs: List[str] = []
        self.div_paragraphs: List[str] = []

        self._in_a = False
        self._current_a_attrs: Dict[str, str] = {}
        self._current_a_text: List[str] = []

        self._in_h1 = False
        self._current_h1_text: List[str] = []

        self._in_title = False
        self._current_title_text: List[str] = []

        self._in_p = False
        self._current_p_text: List[str] = []

        self._div_stack: List[List[str]] = []

        self._in_json_ld = False
        self._current_json_ld: List[str] = []

        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        t = tag.lower()
        attr_dict = {k.lower(): (v or "") for k, v in attrs if k}

        if t in ("script", "style", "nav", "footer", "aside", "noscript", "iframe"):
            if t == "script" and "ld+json" in attr_dict.get("type", "").lower():
                self._in_json_ld = True
                self._current_json_ld = []
                return
            self._ignored_depth += 1
            return

        if t == "link" and attr_dict.get("href"):
            self.head_links.append(attr_dict)
        elif t == "meta":
            self.metas.append(attr_dict)
        elif t == "a" and attr_dict.get("href"):
            self._in_a = True
            self._current_a_attrs = attr_dict
            self._current_a_text = []
        elif t == "h1":
            self._in_h1 = True
            self._current_h1_text = []
        elif t == "title":
            self._in_title = True
            self._current_title_text = []
        elif t == "p" and self._ignored_depth == 0:
            self._in_p = True
            self._current_p_text = []
        elif t == "div" and self._ignored_depth == 0:
            self._div_stack.append([])

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t == "script" and self._in_json_ld:
            self._in_json_ld = False
            raw = "".join(self._current_json_ld).strip()
            if raw:
                self.json_ld_blocks.append(raw)
            return

        if t in ("script", "style", "nav", "footer", "aside", "noscript", "iframe"):
            if self._ignored_depth > 0:
                self._ignored_depth -= 1
            return

        if t == "a" and self._in_a:
            self._in_a = False
            txt = clean_text(" ".join(self._current_a_text))
            rec = dict(self._current_a_attrs)
            rec["text"] = txt
            self.links.append(rec)
        elif t == "h1" and self._in_h1:
            self._in_h1 = False
            txt = clean_text(" ".join(self._current_h1_text))
            if txt:
                self.h1_texts.append(txt)
        elif t == "title" and self._in_title:
            self._in_title = False
            txt = clean_text(" ".join(self._current_title_text))
            if txt:
                self.title_texts.append(txt)
        elif t == "p" and self._in_p:
            self._in_p = False
            txt = clean_text(" ".join(self._current_p_text))
            if txt:
                self.paragraphs.append(txt)
        elif t == "div" and self._div_stack:
            buf = self._div_stack.pop()
            txt = clean_text(" ".join(buf))
            if len(txt) >= 45:
                self.div_paragraphs.append(txt)

    def handle_data(self, data: str) -> None:
        if self._in_json_ld:
            self._current_json_ld.append(data)
            return
        if self._ignored_depth > 0:
            return
        if self._in_a:
            self._current_a_text.append(data)
        if self._in_h1:
            self._current_h1_text.append(data)
        if self._in_title:
            self._current_title_text.append(data)
        if self._in_p:
            self._current_p_text.append(data)
        if self._div_stack and not self._in_a:
            self._div_stack[-1].append(data)


def parse_html_dom(html_content: str) -> _DOMExtractor:
    parser = _DOMExtractor()
    try:
        parser.feed(html_content)
    except Exception:
        pass
    return parser


def get_cairo_now_iso() -> str:
    return datetime.now(CAIRO_TZ).isoformat()


def infer_source_name(url: str, custom_name: Optional[str] = None) -> str:
    if custom_name and custom_name.strip() and custom_name.strip() != "auto":
        return custom_name.strip()
    host = urllib.parse.urlparse(url).netloc.lower()
    host = re.sub(r"^www\.", "", host)
    for domain, ar_name in DOMAIN_NAMES_AR.items():
        if host == domain or host.endswith("." + domain):
            return ar_name
    return host or "مصدر غير معروف"


def normalize_url(url: str) -> str:
    """توحيد الرابط وحذف معاملات التتبع لمنع التكرار (canonical_url)."""
    if not url:
        return ""
    url = url.strip()
    parsed = urllib.parse.urlsplit(url)
    query_params = urllib.parse.parse_qsl(parsed.query, keep_blank_values=False)
    filtered_q = [
        (k, v)
        for k, v in query_params
        if not k.lower().startswith(("utm_", "fbclid", "gclid", "ref"))
    ]
    clean_query = urllib.parse.urlencode(filtered_q)
    clean_path = urllib.parse.quote(urllib.parse.unquote(parsed.path), safe="/%:@")
    return urllib.parse.urlunsplit(
        (parsed.scheme.lower() or "https", parsed.netloc.lower(), clean_path, clean_query, "")
    )


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_url_content(url: str, timeout: int = 20, retries: int = 2) -> Optional[str]:
    """جلب محتوى صفحة أو خلاصة RSS مع التعامل مع الروابط العربية والتكويد وإعادة المحاولة."""
    parsed = urllib.parse.urlsplit(url.strip())
    safe_path = urllib.parse.quote(urllib.parse.unquote(parsed.path), safe="/%:@")
    safe_query = urllib.parse.quote(urllib.parse.unquote(parsed.query), safe="=&?/%:@+")
    safe_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, safe_path, safe_query, ""))

    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(safe_url, headers=DEFAULT_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                final_url = resp.geturl()
                if "404.aspx" in final_url.lower():
                    logger.warning(f"الرابط أعاد التوجيه لصفحة 404: {url} -> {final_url}")
                    return None
                raw_bytes = resp.read()
                for encoding in ("utf-8", "windows-1256", "iso-8859-1"):
                    try:
                        return raw_bytes.decode(encoding)
                    except UnicodeDecodeError:
                        continue
                return raw_bytes.decode("utf-8", errors="ignore")
        except Exception as exc:
            if attempt < retries:
                time.sleep(1.0)
                continue
            logger.warning(f"تعذر جلب الرابط {url}: {exc}")
            return None
    return None


def is_xml_feed(content: str) -> bool:
    head = content[:2500].lstrip().lower()
    return (
        "<rss" in head
        or "<feed" in head
        or "<channel>" in head
        or ("<?xml" in head and ("<item" in head or "<entry" in head or "<channel" in head))
    )


def parse_xml_rss_items(xml_content: str, feed_url: str) -> List[Dict[str, Any]]:
    """استخراج عناصر الأخبار من محتوى XML/RSS/Atom."""
    items: List[Dict[str, Any]] = []
    try:
        cleaned_xml = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", xml_content.lstrip())
        root = ET.fromstring(cleaned_xml)
        for elem in root.iter():
            tag = elem.tag.split("}")[-1].lower()
            if tag in ("item", "entry"):
                title = ""
                link = ""
                summary = ""
                pub_date = ""
                guid = ""
                category = ""
                for child in elem:
                    ctag = child.tag.split("}")[-1].lower()
                    ctext = (child.text or "").strip()
                    if ctag == "title" and not title:
                        title = clean_text(ctext)
                    elif ctag == "link":
                        href = child.attrib.get("href", "").strip()
                        link = href or ctext
                    elif ctag in ("description", "summary", "content", "encoded") and not summary:
                        summary = clean_text(ctext)
                    elif ctag in ("pubdate", "published", "updated", "date") and not pub_date:
                        pub_date = clean_text(ctext)
                    elif ctag in ("guid", "id") and not guid:
                        guid = clean_text(ctext)
                    elif ctag == "category" and not category:
                        category = clean_text(ctext)

                if link:
                    full_link = urllib.parse.urljoin(feed_url, link.strip())
                    items.append(
                        {
                            "item_guid": guid or full_link,
                            "item_url": full_link,
                            "raw_title": title,
                            "raw_summary": summary,
                            "raw_published_at": pub_date,
                            "category_hint": category,
                            "feed_url": feed_url,
                            "discovery_method": "rss_xml",
                        }
                    )
    except Exception:
        item_blocks = re.findall(r"<item\b[^>]*>(.*?)</item>", xml_content, re.S | re.I)
        for block in item_blocks:
            title_m = re.search(r"<title[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", block, re.S | re.I)
            link_m = re.search(r"<link[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</link>", block, re.S | re.I)
            desc_m = re.search(
                r"<description[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</description>", block, re.S | re.I
            )
            pub_m = re.search(r"<pubDate[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</pubDate>", block, re.S | re.I)
            guid_m = re.search(r"<guid[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</guid>", block, re.S | re.I)
            if link_m:
                raw_link = clean_text(link_m.group(1))
                full_link = urllib.parse.urljoin(feed_url, raw_link)
                items.append(
                    {
                        "item_guid": clean_text(guid_m.group(1)) if guid_m else full_link,
                        "item_url": full_link,
                        "raw_title": clean_text(title_m.group(1)) if title_m else "",
                        "raw_summary": clean_text(desc_m.group(1)) if desc_m else "",
                        "raw_published_at": clean_text(pub_m.group(1)) if pub_m else "",
                        "category_hint": "",
                        "feed_url": feed_url,
                        "discovery_method": "rss_regex_fallback",
                    }
                )
    return items


def discover_sub_rss_or_articles_from_html(
    html_content: str, base_url: str, max_sub_feeds: int = 4
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    يكتشف خلاصات RSS الفرعية أو روابط الأخبار المباشرة داخل أي صفحة HTML.
    """
    dom = parse_html_dom(html_content)
    base_host = re.sub(r"^www\.", "", urllib.parse.urlparse(base_url).netloc.lower())

    sub_rss_candidates: List[str] = []
    seen_rss: Set[str] = set()

    for link_tag in dom.head_links:
        ltype = link_tag.get("type", "").lower()
        href = link_tag.get("href", "").strip()
        if href and ("rss" in ltype or "xml" in ltype or "atom" in ltype):
            full_u = urllib.parse.urljoin(base_url, href)
            if full_u not in seen_rss and full_u != base_url:
                seen_rss.add(full_u)
                sub_rss_candidates.append(full_u)

    for a_tag in dom.links:
        href = a_tag.get("href", "").strip()
        if not href or href.startswith(("javascript:", "#", "mailto:")):
            continue
        full_u = urllib.parse.urljoin(base_url, href)
        u_host = re.sub(r"^www\.", "", urllib.parse.urlparse(full_u).netloc.lower())
        if u_host != base_host:
            continue
        low_u = full_u.lower()
        if any(low_u.endswith(ext) for ext in (".css", ".js", ".png", ".jpg", ".ico")):
            continue
        if full_u.rstrip("/") == base_url.rstrip("/"):
            continue
        if (
            "sectionrss" in low_u
            or "/rss/feed/" in low_u
            or low_u.endswith("/rss")
            or "rss.aspx" in low_u
            or "rssfeed" in low_u
        ) and "?page=" not in low_u:
            if full_u not in seen_rss:
                seen_rss.add(full_u)
                sub_rss_candidates.append(full_u)

    direct_articles: List[Dict[str, Any]] = []
    seen_articles: Set[str] = set()

    for a_tag in dom.links:
        href = a_tag.get("href", "").strip()
        if not href or href.startswith(("javascript:", "#", "mailto:")):
            continue
        full_u = urllib.parse.urljoin(base_url, href)
        u_host = re.sub(r"^www\.", "", urllib.parse.urlparse(full_u).netloc.lower())
        if u_host != base_host:
            continue

        path = urllib.parse.unquote(urllib.parse.urlparse(full_u).path)
        if any(
            seg in path.lower()
            for seg in ("/tag/", "/category/", "/section/", "/author/", "/rss", "/login", "/search")
        ):
            continue

        if re.search(r"/\d{4,}", path):
            title_text = a_tag.get("text") or clean_text(a_tag.get("title") or a_tag.get("aria-label") or "")
            norm_u = normalize_url(full_u)
            if norm_u not in seen_articles and len(title_text) >= 12:
                seen_articles.add(norm_u)
                direct_articles.append(
                    {
                        "item_guid": norm_u,
                        "item_url": full_u,
                        "raw_title": title_text,
                        "raw_summary": "",
                        "raw_published_at": "",
                        "category_hint": "",
                        "feed_url": base_url,
                        "discovery_method": "html_page_links",
                    }
                )

    return sub_rss_candidates[:max_sub_feeds], direct_articles


def extract_full_article_details(article_url: str, timeout: int = 20) -> Dict[str, Any]:
    """الدخول على صفحة الخبر واستخراج العنوان، الملخص، النص الكامل، تاريخ النشر، والقسم."""
    details = {
        "title": "",
        "summary": "",
        "article_text": "",
        "published_at": "",
        "category": "",
        "image_url": "",
    }
    html_doc = fetch_url_content(article_url, timeout=timeout)
    if not html_doc:
        return details

    dom = parse_html_dom(html_doc)

    # 1. استخراج البيانات المنظمة JSON-LD إن وجدت
    for raw_json in dom.json_ld_blocks:
        try:
            data = json.loads(raw_json, strict=False)
            candidates = data if isinstance(data, list) else [data]
            for item in candidates:
                if not isinstance(item, dict):
                    continue
                if "@graph" in item and isinstance(item["@graph"], list):
                    candidates.extend(item["@graph"])
                itype = str(item.get("@type", "")).lower()
                if any(k in itype for k in ("newsarticle", "article", "reportage", "webpage")):
                    if not details["title"] and item.get("headline"):
                        details["title"] = clean_text(str(item["headline"]))
                    if not details["summary"] and item.get("description"):
                        details["summary"] = clean_text(str(item["description"]))
                    if not details["article_text"] and item.get("articleBody"):
                        details["article_text"] = clean_text(str(item["articleBody"]))
                    if not details["published_at"] and item.get("datePublished"):
                        details["published_at"] = clean_text(str(item["datePublished"]))
                    if not details["category"] and item.get("articleSection"):
                        sec = item["articleSection"]
                        details["category"] = clean_text(sec[0] if isinstance(sec, list) and sec else str(sec))
        except Exception:
            continue

    # 2. استخراج وسوم Meta القياسية (OpenGraph / Article Meta)
    meta_map = {
        "og:title": "title",
        "twitter:title": "title",
        "og:description": "summary",
        "description": "summary",
        "article:published_time": "published_at",
        "pubdate": "published_at",
        "article:section": "category",
        "og:image": "image_url",
    }
    for meta in dom.metas:
        key = (meta.get("property") or meta.get("name") or "").lower().strip()
        content = meta.get("content", "").strip()
        if key in meta_map and content and not details[meta_map[key]]:
            details[meta_map[key]] = clean_text(content)

    IGNORED_H1 = {
        "الأكثر قراءة",
        "قد يعجبك أيضا",
        "قد يعجبك أيضاً",
        "شارك برأيك",
        "اخبار متعلقة",
        "أخبار متعلقة",
        "أحدث الأخبار",
    }
    if not details["title"] and dom.h1_texts:
        for h1_candidate in dom.h1_texts:
            cleaned_h1 = clean_text(h1_candidate)
            if cleaned_h1 not in IGNORED_H1 and len(cleaned_h1) > 18:
                details["title"] = cleaned_h1
                break

    if not details["title"] and dom.title_texts:
        raw_page_title = clean_text(dom.title_texts[0])
        raw_page_title = re.sub(r"\s*[-|]\s*(بوابة الشروق|اليوم السابع|مصراوي|المصري اليوم|القاهرة 24).*$", "", raw_page_title)
        if len(raw_page_title) > 15:
            details["title"] = raw_page_title

    # 3. استخراج النص الكامل للخبر من الفقرات النصية <p> أو من <div> إذا كان الموقع يضع المتن في div
    if len(details["article_text"]) < 120:
        noise_phrases = (
            "اقرأ أيضا",
            "اقرأ أيضاً",
            "تابعوا آخر أخبار",
            "جميع الحقوق محفوظة",
            "شروط الاستخدام",
        )
        paragraphs: List[str] = []
        candidate_blocks = dom.paragraphs if sum(len(x) for x in dom.paragraphs) >= 120 else dom.div_paragraphs
        for txt in candidate_blocks:
            if len(txt) >= 40 and not any(noise in txt for noise in noise_phrases):
                if txt not in paragraphs:
                    paragraphs.append(txt)
        if paragraphs:
            details["article_text"] = "\n".join(paragraphs)

    return details


def scrape_source(
    source_config: Dict[str, Any],
    max_articles: int = 0,
    fetch_full_text: bool = True,
    delay_seconds: float = 0.2,
    timeout: int = 20,
    run_id: str = "manual_run",
    existing_urls: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    يسحب الأخبار من أي مصدر؛ عند ضبط max_articles=0 يسحب جميع الأخبار المتاحة في اليوم بدون حد أقصى.
    """
    url = source_config.get("url", "").strip()
    source_name = infer_source_name(url, source_config.get("source_name"))
    unlimited = max_articles is None or max_articles <= 0
    logger.info(f"بدء سحب المصدر: {source_name} ({url}) | الحد: {'غير محدود (كل المتاح)' if unlimited else max_articles}")

    content = fetch_url_content(url, timeout=timeout)
    if not content:
        logger.error(f"فشل سحب المصدر: {source_name} ({url})")
        return [], []

    raw_items: List[Dict[str, Any]] = []

    if is_xml_feed(content):
        logger.info(f"الرابط {url} عبارة عن خلاصة XML RSS مباشرة.")
        raw_items.extend(parse_xml_rss_items(content, url))
    else:
        logger.info(f"الرابط {url} عبارة عن صفحة HTML؛ جاري البحث عن خلاصات RSS فرعية أو روابط أخبار...")
        dynamic_sub_feeds = 100 if unlimited else max(4, min(60, (max_articles // 10) + 2))
        sub_feeds, direct_articles = discover_sub_rss_or_articles_from_html(
            content, url, max_sub_feeds=dynamic_sub_feeds
        )

        for sf_url in sub_feeds:
            sf_content = fetch_url_content(sf_url, timeout=timeout)
            if sf_content and is_xml_feed(sf_content):
                parsed_sub = parse_xml_rss_items(sf_content, sf_url)
                logger.info(f"  تم استخراج {len(parsed_sub)} خبر من الخلاصة الفرعية: {sf_url}")
                raw_items.extend(parsed_sub)
                if not unlimited and len(raw_items) >= max_articles * 2:
                    break
            elif sf_content:
                _, sub_direct = discover_sub_rss_or_articles_from_html(sf_content, sf_url, max_sub_feeds=1)
                raw_items.extend(sub_direct)

        if (unlimited or len(raw_items) < max_articles) and direct_articles:
            logger.info(f"  إضافة {len(direct_articles)} خبر مستخرج مباشرة من صفحة الـ HTML.")
            raw_items.extend(direct_articles)

            # دعم تصفح الصفحات التالية (?page=2, 3, ...) إذا كان الموقع يدعم Pagination مثل القاهرة 24
            if "page=" in content.lower():
                sep = "&" if "?" in url else "?"
                max_pages = 35 if unlimited else 15
                for page_num in range(2, max_pages + 1):
                    if not unlimited and len({normalize_url(i["item_url"]) for i in raw_items}) >= max_articles:
                        break
                    paged_url = f"{url.split('?')[0]}{sep}page={page_num}" if "?" not in url else f"{url}&page={page_num}"
                    paged_html = fetch_url_content(paged_url, timeout=timeout)
                    if not paged_html:
                        break
                    _, paged_articles = discover_sub_rss_or_articles_from_html(paged_html, paged_url, max_sub_feeds=0)
                    if not paged_articles:
                        break
                    logger.info(f"  صفحة {page_num}: تم استخراج {len(paged_articles)} خبر من {paged_url}")
                    raw_items.extend(paged_articles)
                    # إذا كانت جميع أخبار هذه الصفحة مسحوبة مسبقًا في الماستر نتوقف هنا (وصلنا لأخبار أمس)
                    if existing_urls and all(normalize_url(a["item_url"]) in existing_urls for a in paged_articles):
                        logger.info(f"  جميع أخبار صفحة {page_num} مسحوبة مسبقًا؛ إيقاف التصفح.")
                        break

    # إذا كان السحب غير محدود أو لم نصل للحد المطلوب، نسحب أيضًا الروابط المباشرة من الصفحة الرئيسية للدومين
    parsed_input = urllib.parse.urlsplit(url)
    root_homepage = f"{parsed_input.scheme}://{parsed_input.netloc}/"
    current_unique_count = len({normalize_url(i["item_url"]) for i in raw_items})
    if root_homepage.rstrip("/") != url.rstrip("/") and (unlimited or current_unique_count < max_articles):
        home_html = fetch_url_content(root_homepage, timeout=timeout)
        if home_html:
            _, home_articles = discover_sub_rss_or_articles_from_html(home_html, root_homepage, max_sub_feeds=0)
            if home_articles:
                logger.info(f"  الصفحة الرئيسية ({root_homepage}): تم اكتشاف {len(home_articles)} خبر من الـ HTML.")
                raw_items.extend(home_articles)

    unique_raw_items: List[Dict[str, Any]] = []
    seen_urls: Set[str] = set()
    for item in raw_items:
        norm_u = normalize_url(item["item_url"])
        if norm_u and norm_u not in seen_urls:
            seen_urls.add(norm_u)
            unique_raw_items.append(item)
        if not unlimited and len(unique_raw_items) >= max_articles:
            break

    now_cairo = get_cairo_now_iso()
    bronze_records: List[Dict[str, Any]] = []
    silver_records: List[Dict[str, Any]] = []

    for idx, item in enumerate(unique_raw_items, start=1):
        fetched_time = get_cairo_now_iso()
        bronze_rec = {
            "source_name": source_name,
            "feed_url": item.get("feed_url", url),
            "item_guid": item.get("item_guid", ""),
            "item_url": item.get("item_url", ""),
            "raw_title": item.get("raw_title", ""),
            "raw_summary": item.get("raw_summary", ""),
            "raw_published_at": item.get("raw_published_at", ""),
            "discovery_method": item.get("discovery_method", ""),
            "fetched_at": fetched_time,
            "ingestion_run_id": run_id,
        }
        bronze_records.append(bronze_rec)

        canonical_u = normalize_url(item["item_url"])
        if existing_urls and canonical_u in existing_urls:
            logger.info(f"  تخطي سحب نص الخبر (موجود مسبقًا في الماستر): {canonical_u}")
            continue

        article_details = {
            "title": "",
            "summary": "",
            "article_text": "",
            "published_at": "",
            "category": "",
            "image_url": "",
        }
        if fetch_full_text and item.get("item_url"):
            article_details = extract_full_article_details(item["item_url"], timeout=timeout)
            if delay_seconds > 0 and idx < len(unique_raw_items):
                time.sleep(delay_seconds)

        final_title = clean_text(article_details["title"] or item.get("raw_title", ""))
        final_summary = clean_text(article_details["summary"] or item.get("raw_summary", ""))
        final_text = article_details["article_text"] or final_summary
        final_pub = article_details["published_at"] or item.get("raw_published_at", "") or now_cairo

        hash_input = f"{source_name}|{final_title}|{final_text[:500]}".encode("utf-8", errors="ignore")
        content_hash = hashlib.sha256(hash_input).hexdigest()

        silver_rec = {
            "source_name": source_name,
            "source_input_url": url,
            "feed_url": item.get("feed_url", url),
            "canonical_url": canonical_u,
            "title": final_title,
            "summary": final_summary,
            "article_text": final_text,
            "category_from_site": article_details["category"] or item.get("category_hint", ""),
            "image_url": article_details["image_url"],
            "published_at": final_pub,
            "fetched_at": fetched_time,
            "language": "ar",
            "content_hash": content_hash,
            "processing_status": "new",
            "ingestion_run_id": run_id,
        }
        silver_records.append(silver_rec)

    logger.info(
        f"اكتمل سحب المصدر [{source_name}]: {len(bronze_records)} سجل خام (Bronze)، و{len(silver_records)} مقال منظف (Silver)."
    )
    return bronze_records, silver_records
