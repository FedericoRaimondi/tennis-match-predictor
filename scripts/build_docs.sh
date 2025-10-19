#!/bin/bash
# Script to build and serve documentation

set -e

ACTION=${1:-serve}

if [ "$ACTION" = "serve" ]; then
    echo "📚 Serving documentation locally..."
    echo "🌐 Documentation will be available at: http://127.0.0.1:8000"
    echo ""
    uv run mkdocs serve
elif [ "$ACTION" = "build" ]; then
    echo "🏗️  Building documentation..."
    uv run mkdocs build
    echo "✅ Documentation built in: site/"
elif [ "$ACTION" = "deploy" ]; then
    echo "🚀 Deploying documentation to GitHub Pages..."
    uv run mkdocs gh-deploy
    echo "✅ Documentation deployed!"
else
    echo "Usage: $0 {serve|build|deploy}"
    exit 1
fi
