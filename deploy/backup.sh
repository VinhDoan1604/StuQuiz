#!/usr/bin/env bash
# Sao lưu Postgres + snapshot Qdrant vào ./backups/<thời điểm>, giữ 7 bản gần nhất.
# Chạy từ thư mục gốc repo. Cron hằng ngày lúc 3h:
#   0 3 * * * cd /opt/studyrag && ./deploy/backup.sh >> backups/backup.log 2>&1
set -euo pipefail

COMPOSE="docker compose -f docker-compose.yml -f docker-compose.prod.yml"
COLLECTION="studyrag_chunks"
DEST="backups/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$DEST"

$COMPOSE exec -T postgres sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' > "$DEST/postgres.dump"

# Image qdrant không có curl, nên gọi API từ container api.
SNAPSHOT=$($COMPOSE exec -T api curl -fsS -X POST "http://qdrant:6333/collections/$COLLECTION/snapshots" \
  | sed -E 's/.*"name":"([^"]+)".*/\1/')
$COMPOSE exec -T api curl -fsS "http://qdrant:6333/collections/$COLLECTION/snapshots/$SNAPSHOT" > "$DEST/qdrant.snapshot"
$COMPOSE exec -T api curl -fsS -X DELETE "http://qdrant:6333/collections/$COLLECTION/snapshots/$SNAPSHOT" > /dev/null

ls -1dt backups/20* | tail -n +8 | xargs -r rm -rf
echo "Đã sao lưu vào $DEST"
