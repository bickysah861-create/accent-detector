#!/usr/bin/env bash
# Push this project to GitHub.
#
# Checks everything that can silently go wrong before touching the remote:
# authentication, the remote URL, a dirty tree, and large files. Then pushes
# and verifies the commit actually landed.
#
# Usage:
#   ./push.sh                 push the current branch
#   ./push.sh "message"       commit any pending changes first, then push

set -uo pipefail
cd "$(dirname "$0")"

# Either spelling of the same repository is fine. An existing remote is left
# alone so an SSH remote is never silently downgraded to HTTPS.
REMOTE_SSH="git@github.com:sakrit204-byte/accentlearning_ml.git"
REMOTE_URL="$REMOTE_SSH"
BRANCH="$(git branch --show-current)"
ok()   { printf '  ok    %s\n' "$1"; }
warn() { printf '  note  %s\n' "$1"; }
die()  { printf '\n  STOP  %s\n\n' "$1"; exit 1; }

echo "Pushing to $(git remote get-url origin 2>/dev/null || echo "$REMOTE_URL")"
echo

# 1. repository present
git rev-parse --git-dir >/dev/null 2>&1 || die "this folder is not a git repository"
ok "git repository found, branch $BRANCH"

# 2. remote configured and correct
if ! git remote get-url origin >/dev/null 2>&1; then
  git remote add origin "$REMOTE_URL"; ok "remote origin added"
else
  current="$(git remote get-url origin)"
  case "$current" in
    *sakrit204-byte/accentlearning_ml*) ok "remote origin correct: $current" ;;
    *) git remote set-url origin "$REMOTE_URL"; ok "remote origin corrected to $REMOTE_URL" ;;
  esac
fi

# 3. commit pending work when a message was supplied
if [ -n "${1:-}" ]; then
  git add -A
  if git diff --cached --quiet; then
    ok "nothing new to commit"
  else
    git commit -q -m "$1" && ok "committed: $1"
  fi
fi

# 4. refuse to push a dirty tree silently
if [ -n "$(git status --porcelain)" ]; then
  git status --short | sed 's/^/        /'
  die "uncommitted changes above. Commit them, or run: ./push.sh \"your message\""
fi
ok "working tree clean"

# 5. warn about files GitHub will reject or complain about
big="$(git ls-files | xargs -r du -k 2>/dev/null | awk '$1 > 50000 {print $2" ("int($1/1024)" MB)"}')"
if [ -n "$big" ]; then
  echo "$big" | sed 's/^/        /'
  warn "files above 50 MB. GitHub warns past 50 MB and rejects past 100 MB"
fi
ok "$(git ls-files | wc -l) files tracked, $(git ls-files | xargs -r du -ch 2>/dev/null | tail -1 | cut -f1) total"

# 6. authentication. Either an SSH key or the gh CLI will do.
AUTH=""
case "$(git remote get-url origin)" in
  git@*|ssh://*)
    who="$(GIT_SSH_COMMAND="${SSH_CMD:-$(git config core.sshCommand || echo ssh)}" \
           ssh -o BatchMode=yes -T git@github.com 2>&1 | sed -n 's/^Hi \([^!]*\)!.*/\1/p')"
    [ -n "$who" ] && AUTH="ssh key, as $who" ;;
esac
if [ -z "$AUTH" ] && gh auth status >/dev/null 2>&1; then
  AUTH="gh cli, as $(gh api user --jq .login 2>/dev/null || echo 'unknown user')"
fi
if [ -z "$AUTH" ]; then
  cat <<'MSG'

  STOP  No GitHub authentication, so the push cannot proceed. Pick one.

        A. SSH key, if you already have one on this account:

               git remote set-url origin git@github.com:sakrit204-byte/accentlearning_ml.git
               ssh -T git@github.com          # should greet you by username

           For a key with a custom filename, point this repository at it:

               git config core.sshCommand "ssh -i ~/.ssh/YOUR_KEY -o IdentitiesOnly=yes"

        B. GitHub CLI:

               gh auth login
               # GitHub.com  ->  HTTPS  ->  authenticate with browser

MSG
  exit 1
fi
ok "authenticated by $AUTH"

# 7. push
echo
echo "pushing $(git log --oneline | wc -l) commits on $BRANCH"
if git push -u origin "$BRANCH"; then
  echo
  remote_head="$(git ls-remote origin "refs/heads/$BRANCH" 2>/dev/null | cut -f1)"
  local_head="$(git rev-parse HEAD)"
  if [ "$remote_head" = "$local_head" ]; then
    ok "verified: remote $BRANCH is at $(git rev-parse --short HEAD)"
    echo
    echo "  https://github.com/sakrit204-byte/accentlearning_ml"
  else
    die "push reported success but the remote head does not match. Check manually."
  fi
else
  die "push failed. See the git output above."
fi
