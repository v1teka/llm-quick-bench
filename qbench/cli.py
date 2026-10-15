"""Командная строка: python -m qbench <команда>."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

from . import checks as K
from . import prompts
from .llm import NeedResponse, make_client
from .runner import Run, load_run, save_run, slug, update_leaderboard
from .scene import Scene, parse_response


def cmd_run(a):
    label = a.label or (a.model if not a.model.startswith("mock") else f"Mock ({a.model.split(':')[-1]})")
    if a.manual:
        out = a.out_dir or a.manual
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out = a.out_dir or os.path.join(a.results, slug(label), stamp)
    os.makedirs(out, exist_ok=True)
    extra = json.loads(a.extra) if a.extra else None
    client = make_client(a.model, base_url=a.base_url, api_key=os.environ.get(a.api_key_env) if a.api_key_env else None,
                         max_tokens=a.max_tokens, temperature=a.temperature, extra=extra,
                         manual=os.path.join(a.manual, "dialog") if a.manual else None)
    print(f"Прогон «{label}» → {out}")
    run = Run(client, label, out, fix_rounds=a.fix_rounds, change=not a.no_change,
              change_fix_rounds=a.change_fix_rounds, do_vision=a.vision, quality=a.quality)
    try:
        summary = run.execute()
    except NeedResponse as e:
        print("\nРучной режим: отдайте модели промпт и сохраните её ответ, затем запустите команду ещё раз.")
        print(f"  промпт: {e.prompt_path}\n  ответ:  {e.response_path}")
        sys.exit(3)
    save_run(summary, out)
    print(f"\nИтог: {summary['overall']:.0%}")
    for k, v in summary["abilities"].items():
        print(f"  {k:10s} {'—' if v is None else f'{v:.0%}'}")
    if not a.no_video:
        from .render.video import make_video
        print("Рендер видео…")
        paths = make_video(summary, out, quality=a.quality, gif=True, mp4=not a.no_mp4, gif_width=a.gif_width)
        for k, p in paths.items():
            print(f"  {k}: {p}")
    print(f"Лидерборд: {update_leaderboard(a.results)}")


def cmd_video(a):
    from .render.video import make_video
    data = load_run(a.run_dir)
    paths = make_video(data, a.run_dir, quality=a.quality, mp4=not a.no_mp4, gif_width=a.gif_width)
    for k, p in paths.items():
        print(f"{k}: {p}")


def cmd_validate(a):
    with open(a.scene, encoding="utf-8") as f:
        pr = parse_response(f.read())
    if a.prev:
        prev = parse_response(open(a.prev, encoding="utf-8").read())
        b0 = K.run_base(prev)
        base, extra = K.run_change(pr, prev.scene, b0)
        cs = base + extra
    else:
        cs = K.run_base(pr)
    for c in cs:
        print(f"{'✓' if c.passed else '✗'} {c.id:4s} {c.name:28s} {c.score:4.0%}  {c.detail}")
    print(f"\nВзвешенно: {K.weighted(cs):.0%}")


def cmd_render(a):
    from .render.video import render_still
    sc = Scene.load(a.scene)
    render_still(sc, a.output, azim=a.azim, elev=a.elev, quality=a.quality)
    print(a.output)


def cmd_prompt(a):
    print(prompts.SYSTEM + "\n\n" + prompts.BUILD)


def cmd_leaderboard(a):
    print(update_leaderboard(a.results))


def cmd_hard(a):
    from .hard import runner as HR
    from .hard import film
    label = a.label or (a.model if a.model != "mock" else "Mock (Pro демо)")
    if a.manual:
        out = a.out_dir or a.manual
        factory = HR.manual_factory(a.manual)
    elif a.model == "mock":
        out = a.out_dir or os.path.join(a.results, "pro", "mock")
        factory = HR.mock_factory()
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out = a.out_dir or os.path.join(a.results, "pro", slug(label), stamp)
        extra = json.loads(a.extra) if a.extra else None
        factory = HR.api_factory(a.model, base_url=a.base_url,
                                 api_key=os.environ.get(a.api_key_env) if a.api_key_env else None,
                                 max_tokens=a.max_tokens, temperature=a.temperature, extra=extra,
                                 overrun_retries=a.overrun_retries)
    os.makedirs(out, exist_ok=True)
    print(f"Прогон Pro «{label}» → {out}")
    run = HR.HardRun(factory, label, out, fix_rounds=a.fix_rounds)
    try:
        summary = run.execute()
    except HR.Pending as e:
        print("\nРучной режим: нужны ответы модели на промпты (их можно решать параллельно):")
        for p in e.args[0]:
            print(f"  промпт: {p.prompt_path}\n  ответ:  {p.response_path}")
        sys.exit(3)
    HR.save_summary(summary, out)
    print(f"\nИтог: {summary['overall']:.0%}")
    for k, v in summary["abilities"].items():
        print(f"  {k:12s} {'—' if v is None else f'{v:.0%}'}")
    if not a.no_video:
        scene = next((p["scene"] for p in reversed(summary["phases"]) if p["scene"] is not None), None)
        if scene is None:
            print("Сцены дома нет — ролик не собран")
            return
        print("Рендер ролика…")
        frames = film.build_film(summary, scene, summary["tasks"], out, quality=a.quality, fps=a.fps)
        paths = film.save_outputs(frames, out, fps=a.fps)
        frames[len(frames) // 3].save(os.path.join(out, "poster.png"))
        for k, p in paths.items():
            print(f"  {k}: {p}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="qbench", description="llm-quick-bench · Дом-3D: быстрый комплексный бенчмарк LLM с видео-результатом")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="прогнать модель")
    r.add_argument("--model", required=True, help="id модели у провайдера, либо mock:flawed | mock:ideal | mock:broken")
    r.add_argument("--label", help="подпись в видео и лидерборде")
    r.add_argument("--base-url", help="OpenAI-совместимый endpoint (по умолчанию OpenRouter или $QBENCH_BASE_URL)")
    r.add_argument("--api-key-env", help="имя переменной окружения с ключом (по умолчанию QBENCH_API_KEY / OPENROUTER_API_KEY / OPENAI_API_KEY)")
    r.add_argument("--max-tokens", type=int, default=32000, help="0 — не передавать")
    r.add_argument("--temperature", type=float)
    r.add_argument("--extra", help='доп. поля запроса JSON, напр. \'{"reasoning": {"effort": "high"}}\'')
    r.add_argument("--fix-rounds", type=int, default=1, help="раундов самокоррекции (по умолчанию 1)")
    r.add_argument("--change-fix-rounds", type=int, default=0)
    r.add_argument("--no-change", action="store_true", help="без фазы «гараж»")
    r.add_argument("--vision", action="store_true", help="добавить визуальный тест (нужна мультимодальная модель)")
    r.add_argument("--manual", metavar="DIR", help="ручной режим: промпты и ответы — файлы в DIR/dialog")
    r.add_argument("--quality", choices=["fast", "normal", "high"], default="normal")
    r.add_argument("--gif-width", type=int, help="ширина GIF (по умолчанию как у видео)")
    r.add_argument("--no-video", action="store_true")
    r.add_argument("--no-mp4", action="store_true")
    r.add_argument("--results", default="results")
    r.add_argument("--out-dir")
    r.set_defaults(fn=cmd_run)

    d = sub.add_parser("demo", help="демо-прогон на mock-модели (без API)")
    d.set_defaults(fn=lambda a: cmd_run(argparse.Namespace(
        model="mock:flawed", label="Mock (демо)", base_url=None, api_key_env=None, max_tokens=0, temperature=None,
        extra=None, fix_rounds=1, change_fix_rounds=0, no_change=False, vision=True, manual=None, quality=a.quality,
        gif_width=None, no_video=False, no_mp4=False, results=a.results, out_dir=a.out_dir)))
    d.add_argument("--quality", choices=["fast", "normal", "high"], default="normal")
    d.add_argument("--results", default="results")
    d.add_argument("--out-dir")

    v = sub.add_parser("video", help="перерендерить видео сохранённого прогона")
    v.add_argument("run_dir")
    v.add_argument("--quality", choices=["fast", "normal", "high"], default="normal")
    v.add_argument("--gif-width", type=int)
    v.add_argument("--no-mp4", action="store_true")
    v.set_defaults(fn=cmd_video)

    c = sub.add_parser("validate", help="проверить JSON-сцену (или ответ модели)")
    c.add_argument("scene")
    c.add_argument("--prev", help="сцена до изменения ТЗ — включает проверки гаража")
    c.set_defaults(fn=cmd_validate)

    s = sub.add_parser("render", help="кадр сцены в PNG")
    s.add_argument("scene")
    s.add_argument("-o", "--output", default="scene.png")
    s.add_argument("--azim", type=float, default=225)
    s.add_argument("--elev", type=float, default=22)
    s.add_argument("--quality", choices=["fast", "normal", "high"], default="normal")
    s.set_defaults(fn=cmd_render)

    pp = sub.add_parser("prompt", help="напечатать задание (для ручного прогона в чате)")
    pp.set_defaults(fn=cmd_prompt)

    h = sub.add_parser("pro", help="жёсткий набор «Дом-3D Pro»: детальный дом + 4 задачи на телевизорах")
    h.add_argument("--model", required=True, help="id модели у провайдера или mock")
    h.add_argument("--label")
    h.add_argument("--base-url")
    h.add_argument("--api-key-env")
    h.add_argument("--max-tokens", type=int, default=128000, help="лимит вывода на ответ (рассуждения входят в него)")
    h.add_argument("--overrun-retries", type=int, default=3,
                   help="сколько раз повторить запрос, если лимит исчерпан до ответа (как в Claude Code)")
    h.add_argument("--temperature", type=float)
    h.add_argument("--extra")
    h.add_argument("--fix-rounds", type=int, default=1)
    h.add_argument("--manual", metavar="DIR")
    h.add_argument("--quality", choices=["fast", "normal", "high"], default="normal")
    h.add_argument("--fps", type=int, default=12)
    h.add_argument("--no-video", action="store_true")
    h.add_argument("--results", default="results")
    h.add_argument("--out-dir")
    h.set_defaults(fn=cmd_hard)

    lb = sub.add_parser("leaderboard", help="пересобрать results/LEADERBOARD.md")
    lb.add_argument("--results", default="results")
    lb.set_defaults(fn=cmd_leaderboard)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
