#!/usr/bin/env bash
# TING — attention signal khi Claude chờ user.
# Usage: ting.sh {askq|notify}
#   askq   = PreToolUse AskUserQuestion/ExitPlanMode (fire TRƯỚC khi câu hỏi hiện ra)
#   notify = Notification permission_prompt/idle_prompt/agent_needs_input
#
# Nguyên tắc: 1 wait state ≈ 1 alert (dedup 5s); fail safely (luôn exit 0, không block tool);
# không secrets, không log ngoài stamp file.
#
# LIMITATION (đã biết, không claim ngược lại): khi SSH disconnected mà tmux remote vẫn sống,
# BEL/tmux message KHÔNG tới được laptop. Muốn notify lúc detached cần external push
# transport (webhook/push service) — chưa cài, chỉ thêm khi user yêu cầu. Chừa chỗ ở cuối file.

EVENT="${1:-notify}"
STAMP="${TMPDIR:-/tmp}/claude_ting_stamp"

now=$(date +%s 2>/dev/null) || now=0
last=0
[ -f "$STAMP" ] && last=$(cat "$STAMP" 2>/dev/null || echo 0)
case "$last" in ''|*[!0-9]*) last=0;; esac
if [ "$now" -gt 0 ] && [ $((now - last)) -lt 5 ]; then
    exit 0
fi

fired=0
case "$(uname -s 2>/dev/null)" in
    MINGW*|MSYS*|CYGWIN*)
        # Windows local: ~/.claude/settings.json (user-level) ĐÃ phát âm thanh cho
        # Notification + Stop → ở đây chỉ kêu cho askq để tránh double-fire.
        if [ "$EVENT" = "askq" ]; then
            (powershell.exe -NoProfile -Command "[console]::beep(880,200); [console]::beep(1175,250)" >/dev/null 2>&1 &)
            fired=1
        fi
        ;;
    *)
        # Linux/remote (Vast, trong tmux qua SSH): BEL tới client đang attach.
        if [ -n "$TMUX" ] && command -v tmux >/dev/null 2>&1; then
            tmux display-message "Claude cho y ban (${EVENT})" 2>/dev/null || true
        fi
        # BEL: uu tien tty that; fallback stdout.
        { printf '\a' > /dev/tty; } 2>/dev/null || printf '\a' 2>/dev/null || true
        fired=1
        # [EXTENSION POINT] external push transport khi SSH detached, ví dụ:
        #   curl -m 3 -s -X POST "$TING_WEBHOOK_URL" -d "event=${EVENT}" || true
        # Chỉ bật khi user yêu cầu và TING_WEBHOOK_URL được cấp qua env (không hard-code secret).
        ;;
esac

[ "$fired" = "1" ] && { echo "$now" > "$STAMP" 2>/dev/null || true; }
exit 0
