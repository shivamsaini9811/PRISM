"""Exercise the browser transport against a fake chat site served by Playwright routing."""

import asyncio

import pytest

from prism.browser import (BotCheck, BrowserManager, BrowserProvider, NotLoggedIn, PageNotReady,
                           SiteAdapter, SiteConfig)
from prism.browser.adapter import clean_reply

FAKE_CHAT = """<!doctype html><html><body>
<nav><textarea id="search" placeholder="Search chats" style="width:90px;height:18px"></textarea></nav>
<div id="log"></div>
<textarea id="composer"></textarea><input type="file" id="upload" multiple hidden><button id="send">Send</button>
<script>
if (location.hash === '#late') setTimeout(() => {  // the real message box loads slowly
  document.getElementById('composer').id = 'late';
}, 6500);
if (location.hash === '#input') {  // a message box that isn't a text area: nothing generic finds it
  const i = document.createElement('input'); i.type = 'text'; i.id = 'composer';
  i.style.cssText = 'width:600px;height:40px'; document.getElementById('composer').replaceWith(i);
}
let n = 0;
const remount = location.hash === '#remount';  // read now: pushState drops the hash
const delay = location.hash === '#slow' ? 3000 : 400, virtual = location.hash === '#virtual';  // #slow: the model thinks first
document.getElementById('composer').onkeydown = e => {  // Enter sends, like real chat sites
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); document.getElementById('send').click(); }
};
document.getElementById('send').onclick = () => {
  const box = document.querySelector('#composer, #late'), q = box.value; box.value = '';
  if (!location.pathname.startsWith('/c/')) history.pushState({}, '', '/c/42');
  if (remount) log.querySelectorAll('.assistant').forEach(e => e.replaceWith(e.cloneNode(true)));  // React re-draws old replies
  const u = document.createElement('div'); u.className = 'user'; u.textContent = q; log.appendChild(u);
  setTimeout(() => {                       // the reply box shows up a moment after the question
  const a = document.createElement('div'); a.className = 'assistant'; log.appendChild(a);
  if (virtual) { const old = log.querySelectorAll('.assistant'); if (old.length > 1) old[0].remove(); }  // long-chat view
  const spin = document.createElement('span'); spin.className = 'streaming'; document.body.appendChild(spin);
  const up = document.getElementById('upload'), got = [...up.files].map(f => f.name); up.value = '';
  const words = ('echo: ' + q + (got.length ? ' [files: ' + got.join(', ') + ']' : '')).split(' '); let i = 0;
  const t = setInterval(() => {            // stream the reply word by word
    a.textContent += (i ? ' ' : '') + words[i++];
    if (i === words.length) {
      clearInterval(t); spin.remove();
      if (q.includes('picture')) {  // a generated image and a file to download
        const c = document.createElement('canvas'); c.width = c.height = 120;
        c.getContext('2d').fillRect(0, 0, 120, 120);
        const img = document.createElement('img'); img.src = c.toDataURL('image/png'); a.appendChild(img);
        const icon = document.createElement('img'); icon.src = img.src; icon.width = icon.height = 16; a.appendChild(icon);
        const dl = document.createElement('a'); dl.download = 'plan.txt'; dl.href = 'data:text/plain,the plan';
        dl.textContent = 'Download'; a.appendChild(dl);
      }
    }
  }, 150);
  }, delay);
};
if (location.hash.startsWith('#cookies')) {  // grok.com: a cookie notice over the message box
  const b = document.createElement('div'); b.setAttribute('data-cookie-banner', '');
  b.style.cssText = 'position:fixed;inset:0;z-index:9;background:rgba(255,255,255,.5)';
  b.innerHTML = '<p>We use cookies</p>' + (location.hash === '#cookies' ? '<button>Reject All</button>' : '');
  b.onclick = e => { if (e.target.tagName === 'BUTTON') b.remove(); };
  document.body.appendChild(b);
}
</script></body></html>"""

LOGIN_PAGE = "<!doctype html><html><body><h1>Please sign in</h1></body></html>"
# claude.ai style: the app loads, then its script sends you to the login page.
REDIRECT_PAGE = ("<!doctype html><html><body><script>setTimeout(() => "
                 "location.href = 'http://fake.test/login', 500)</script></body></html>")
BOT_PAGE = "<!doctype html><html><head><title>Just a moment...</title></head><body></body></html>"
ANON_PAGE = ("<!doctype html><html><body><button>Log in</button>"
             "<textarea id='composer'></textarea></body></html>")
BLANK_PAGE = "<!doctype html><html><body><p>still loading</p></body></html>"

CONFIG = SiteConfig(new_chat_url="http://fake.test/new", composer="#composer",
                    assistant_message=".assistant", send_button="#send", streaming=".streaming")


@pytest.fixture
async def manager(tmp_path):
    m = BrowserManager(tmp_path / "profiles", headless=True)

    async def serve(route):
        url = route.request.url
        body = (LOGIN_PAGE if "login" in url else REDIRECT_PAGE if "later" in url else
                ANON_PAGE if "anon" in url else BOT_PAGE if "bot" in url
                else BLANK_PAGE if "blank" in url else FAKE_CHAT)
        await route.fulfill(status=200, content_type="text/html", body=body)

    m.serve = serve
    try:
        await (await m.context()).route("http://fake.test/**", serve)  # one browser, all platforms
    except Exception as e:
        pytest.skip(f"no browser available: {e}")
    yield m
    await m.close()


async def test_daughter_binds_to_remote_thread_and_reads_streamed_reply(manager):
    p = BrowserProvider("fake", manager, SiteAdapter(CONFIG, load_timeout_s=5))
    reply = await p.complete([{"role": "user", "content": "hello there"}], system="SYS")
    # First message of a new remote chat carries the system prompt.
    assert reply.startswith("echo: SYS") and reply.endswith("hello there")
    assert p.thread_url == "http://fake.test/c/42"

    # Second turn continues the bound thread and does not repeat the system prompt.
    reply2 = await p.complete([{"role": "user", "content": "hello there"},
                               {"role": "assistant", "content": reply},
                               {"role": "user", "content": "again"}], system="SYS")
    assert reply2 == "echo: again"


async def test_fresh_mother_always_starts_new_chat(manager):
    p = BrowserProvider("fake", manager, SiteAdapter(CONFIG, load_timeout_s=5), fresh=True)
    assert (await p.complete([{"role": "user", "content": "plan"}], system="S")).endswith("plan")
    assert p.thread_url is None


@pytest.mark.parametrize("path,error", [("login", NotLoggedIn), ("bot", BotCheck),
                                        ("blank", PageNotReady)])
async def test_missing_composer_is_diagnosed_not_assumed_logged_out(manager, tmp_path, path, error):
    cfg = CONFIG.with_overrides({"new_chat_url": f"http://fake.test/{path}"})
    adapter = SiteAdapter(cfg, load_timeout_s=1, screenshot_dir=tmp_path / "shots")
    p = BrowserProvider("fake", manager, adapter)
    with pytest.raises(error) as e:
        await p.complete([{"role": "user", "content": "hi"}])
    assert type(e.value) is error and "screenshot:" in str(e.value)
    assert list((tmp_path / "shots").glob("*.png"))


async def test_redirect_to_login_fails_fast(manager):
    cfg = CONFIG.with_overrides({"new_chat_url": "http://fake.test/later"})
    p = BrowserProvider("fake", manager, SiteAdapter(cfg, load_timeout_s=60))
    start = asyncio.get_running_loop().time()
    with pytest.raises(NotLoggedIn):
        await p.complete([{"role": "user", "content": "hi"}])
    assert asyncio.get_running_loop().time() - start < 10


async def test_echo_of_own_message_is_not_taken_as_the_reply(manager):
    # Grok's bubbles: the reply selector also matches the user's message.
    cfg = CONFIG.with_overrides({"assistant_message": ".user, .assistant",
                                 "new_chat_url": "http://fake.test/new#slow"})
    p = BrowserProvider("fake", manager, SiteAdapter(cfg, load_timeout_s=5))
    prompt = "a long enough question that the site shows back to us"
    assert await p.complete([{"role": "user", "content": prompt}]) == "echo: " + prompt


async def test_new_reply_found_when_old_ones_leave_the_page(manager):
    # Two turns on one page; the oldest reply leaves as the new one arrives, so the number of
    # replies on the page stays the same and counting them would wait forever.
    adapter = SiteAdapter(CONFIG.with_overrides({"new_chat_url": "http://fake.test/new#virtual"}),
                          load_timeout_s=5)
    page = await manager.new_page()
    await adapter.open(page)
    assert await adapter.send(page, "one", timeout_s=20) == "echo: one"
    assert await asyncio.wait_for(adapter.send(page, "two", timeout_s=20), 30) == "echo: two"


@pytest.mark.parametrize("hash_", ["#cookies", "#cookies-stuck"])
async def test_cookie_notice_over_the_message_box(manager, hash_):
    # #cookies: refused via its button; #cookies-stuck: can't be closed, so type without clicking
    cfg = CONFIG.with_overrides({"new_chat_url": f"http://fake.test/new{hash_}"})
    p = BrowserProvider("fake", manager, SiteAdapter(cfg, load_timeout_s=10))
    assert await asyncio.wait_for(p.complete([{"role": "user", "content": "hi"}]), 30) == "echo: hi"


async def test_files_go_up_and_pictures_and_files_come_back(manager, tmp_path):
    media = tmp_path / "media"
    note = tmp_path / "brief.txt"
    note.write_text("brief")
    p = BrowserProvider("fake", manager, SiteAdapter(CONFIG, load_timeout_s=10, media_dir=media))
    reply = await p.complete_with_files([{"role": "user", "content": "draw a picture"}], None, [str(note)])
    assert reply.startswith("echo: draw a picture [files: brief.txt]")
    lines = reply.splitlines()
    image = next(x for x in lines if x.startswith("![image](media/"))
    file = next(x for x in lines if x.startswith("[\U0001F4C4 plan.txt](media/"))
    assert len([x for x in lines if x.startswith("![image]")]) == 1  # the 16px icon is skipped
    saved = {f.name: f for f in media.iterdir()}
    assert saved[image[15:-1]].read_bytes().startswith(b"\x89PNG")
    assert saved[file.split("(media/")[1][:-1]].read_text() == "the plan"


async def test_renamed_message_box_and_send_button_still_work(manager):
    # The site changed its layout: the configured selectors match nothing any more.
    cfg = CONFIG.with_overrides({"composer": "#prompt-textarea", "send_button": "#send-button"})
    p = BrowserProvider("fake", manager, SiteAdapter(cfg, load_timeout_s=10))
    assert await p.complete([{"role": "user", "content": "hello"}]) == "echo: hello"


async def test_signed_out_page_with_a_chat_box_is_not_used(manager):
    cfg = CONFIG.with_overrides({"new_chat_url": "http://fake.test/anon",
                                 "signed_out": 'button:has-text("Log in")'})
    p = BrowserProvider("fake", manager, SiteAdapter(cfg, load_timeout_s=30))
    with pytest.raises(NotLoggedIn):
        await p.complete([{"role": "user", "content": "hi"}])
    page = await manager.new_page()
    report = await SiteAdapter(cfg).doctor(page)
    assert report["message_box"] == "textarea#composer" and not report["logged_in"]


async def test_platforms_run_in_parallel(manager):
    providers = [BrowserProvider(p, manager, SiteAdapter(CONFIG, load_timeout_s=10))
                 for p in ("fake", "fake2", "fake3")]
    replies = await asyncio.gather(*(p.complete([{"role": "user", "content": f"q{i}"}])
                                     for i, p in enumerate(providers)))
    assert replies == ["echo: q0", "echo: q1", "echo: q2"]


def test_clean_reply_drops_claude_label_thinking_summary_and_doc_button():
    # Copied from a real claude.ai reply.
    raw = ("Claude responded: Agreement\nWeighing minor leftover differences.\n\n"
           "Weighing minor leftover differences.\nAgreement\n\nClaude and Gemini merged.\n\n"
           "STATUS: converged\n\n\nCreate a doc")
    assert clean_reply(raw) == "Agreement\n\nClaude and Gemini merged.\n\nSTATUS: converged"
    assert clean_reply("The model responded: well") == "well"
    assert clean_reply("Grok\n\nRed\nBlue") == "Red\nBlue"  # grok.com's heading line
    assert clean_reply("Grok\nWorked for 4s\nMorning (daily)\nCleanser") == "Morning (daily)\nCleanser"
    # claude.ai without the label: preview line, tool notes, then the answer (copied from a real reply)
    raw = ("Yes - from a past chat on 16 Oct.\nUsed Search Past Conversations, searched past chats\n"
           "Used Search Past Conversations, searched past chats\nYes - from a past chat on 16 Oct. Note: x.\nMORNING")
    assert clean_reply(raw) == "Yes - from a past chat on 16 Oct. Note: x.\nMORNING"
    assert clean_reply("Step 1\n" + "long text " * 30 + "\nStep 1 again") .startswith("Step 1\nlong")
    assert clean_reply("Grok is a model.") == "Grok is a model."


def test_clean_reply_strips_icons_and_page_chrome():
    raw = "Step 1: basics\nLearn the timing.\n\n\ue001\n\u25a1\n\u25a1\nCopy\njust now\n"
    assert clean_reply(raw) == "Step 1: basics\nLearn the timing."
    assert clean_reply("5 minutes ago is fine\nok") == "5 minutes ago is fine\nok"


async def test_doctor_reports_selector_matches(manager):
    page = await manager.new_page()
    report = await SiteAdapter(CONFIG).doctor(page)
    assert report["logged_in"] and report["checks"]["send_button"]["matches"] == 1
    assert report["checks"]["assistant_message"]["matches"] == 0


async def test_platforms_share_one_browser(manager):
    ctx = await manager.context()
    providers = [BrowserProvider(p, manager, SiteAdapter(CONFIG, load_timeout_s=10))
                 for p in ("fake", "fake2", "fake3")]
    await asyncio.gather(*(p.complete([{"role": "user", "content": "q"}]) for p in providers))
    assert await manager.context() is ctx


async def test_recovers_after_the_browser_dies(manager):
    await manager._pw.stop()  # what an out-of-memory kill of the driver looks like to PRISM
    page = await manager.new_page()  # relaunches instead of failing until restart
    await page.context.route("http://fake.test/**", manager.serve)
    await page.close()
    p = BrowserProvider("fake", manager, SiteAdapter(CONFIG, load_timeout_s=10))
    assert await p.complete([{"role": "user", "content": "back"}]) == "echo: back"


async def test_renamed_reply_is_read_by_watching_the_page_and_learned(manager):
    # The site renamed its reply blocks: the configured selector never matches.
    learned = []
    cfg = CONFIG.with_overrides({"assistant_message": ".bot-answer-v2", "streaming": ".gone"})
    adapter = SiteAdapter(cfg, load_timeout_s=5, on_learn=learned.append, fallback_after_s=1)
    p = BrowserProvider("fake", manager, adapter)
    reply = await asyncio.wait_for(p.complete([{"role": "user", "content": "where did it go"}]), 30)
    assert reply == "echo: where did it go"
    assert learned == [{"assistant_message": "div.assistant"}]
    assert p.notes and "reply layout changed" in p.notes[0]
    # Next turn uses what was learned, straight away.
    reply2 = await asyncio.wait_for(p.complete([{"role": "user", "content": "where did it go"},
                                                {"role": "assistant", "content": reply},
                                                {"role": "user", "content": "and now"}]), 30)
    assert reply2 == "echo: and now" and p.notes == []


async def test_renamed_message_box_is_found_not_the_sidebar_search(manager):
    learned = []
    cfg = CONFIG.with_overrides({"composer": "#prompt-v2"})
    p = BrowserProvider("fake", manager, SiteAdapter(cfg, load_timeout_s=15, on_learn=learned.append))
    assert await p.complete([{"role": "user", "content": "hello box"}]) == "echo: hello box"
    assert learned == [{"composer": "textarea#composer"}]


async def test_another_assistant_finds_an_unusual_message_box(manager):
    questions = []

    async def advisor(question):
        questions.append(question)
        line = next(x for x in question.splitlines() if "tag=input" in x)
        return line.split()[0].removeprefix("n=")

    cfg = CONFIG.with_overrides({"composer": "#prompt-v2", "new_chat_url": "http://fake.test/new#input"})
    p = BrowserProvider("fake", manager, SiteAdapter(cfg, load_timeout_s=3, advisor=advisor))
    assert await p.complete([{"role": "user", "content": "odd box"}]) == "echo: odd box"
    assert len(questions) == 1 and "odd box" not in questions[0]  # no chat text goes out
    assert any("another assistant" in n for n in p.notes)


async def test_watching_the_page_ignores_old_replies_it_redraws(manager):
    # Two turns on one page; before the second reply the page re-draws the first one.
    cfg = CONFIG.with_overrides({"assistant_message": ".bot-answer-v2", "streaming": ".gone",
                                 "new_chat_url": "http://fake.test/new#remount"})
    page = await manager.new_page()
    try:
        first = SiteAdapter(cfg, load_timeout_s=5, fallback_after_s=1)
        await first.open(page)
        assert await asyncio.wait_for(first.send(page, "a first question with a long answer"), 30) \
            == "echo: a first question with a long answer"
        fresh = SiteAdapter(cfg, load_timeout_s=5, fallback_after_s=1)  # has learned nothing
        assert await asyncio.wait_for(fresh.send(page, "second"), 30) == "echo: second"
    finally:
        await page.close()


async def test_slow_message_box_is_not_learned_as_a_new_one(manager):
    learned = []
    cfg = CONFIG.with_overrides({"composer": "#late", "new_chat_url": "http://fake.test/new#late"})
    p = BrowserProvider("fake", manager, SiteAdapter(cfg, load_timeout_s=20, on_learn=learned.append))
    assert await p.complete([{"role": "user", "content": "patience"}]) == "echo: patience"
    assert learned == []
