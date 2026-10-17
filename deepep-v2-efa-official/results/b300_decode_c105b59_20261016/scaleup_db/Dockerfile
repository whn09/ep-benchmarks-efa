# Experiment-only: EP_COMBINE_SCALEUP_DB (combine scale-up double buffering), layered
# on either the chcap-cb image (clean timing) or the chcap-tl-cb image (phase breakdown).
# TIMELINE_CHECK=1 additionally asserts the timeline header survived the reinstall.
ARG BASE
FROM ${BASE}
ARG TIMELINE_CHECK=0
COPY apply_db.py /tmp/db/
RUN set -eux; cd /opt/DeepEP; \
    python3 /tmp/db/apply_db.py /opt/DeepEP
RUN set -eux; cd /opt/DeepEP; \
    SITE="$(cd / && python3 -c 'import importlib.util,os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("deep_ep").origin)))')"; \
    pip uninstall -y deep-ep deep_ep 2>/dev/null || true; rm -rf "$SITE"/deep_ep*; \
    if [ -f "$SITE/easy-install.pth" ]; then grep -v deep_ep "$SITE/easy-install.pth" > /tmp/pth || true; mv /tmp/pth "$SITE/easy-install.pth"; fi; \
    NCCL_LIB="$(python3 -c 'import nvidia.nccl, os; print(os.path.join(list(nvidia.nccl.__path__)[0], "lib"))')"; \
    LIBRARY_PATH="$NCCL_LIB:/usr/local/cuda/lib64/stubs" python3 setup.py install; rm -rf build; \
    PKG="$(cd / && python3 -c 'import importlib.util,os; print(os.path.dirname(importlib.util.find_spec("deep_ep").origin))')"; \
    if [ "$TIMELINE_CHECK" = 1 ]; then grep -q '#define EP_TIMELINE 1' "$(find "$PKG" -name timeline.cuh | head -1)"; fi; \
    strings "$(find "$PKG" -maxdepth 1 -name '_C*.so' | head -1)" | grep -q EP_COMBINE_SCALEUP_DB; \
    grep -q 'kScaleupDB ? 1 : 0' "$(find "$PKG" -name hybrid_combine_unordered.cuh | head -1)"; \
    echo "scaleup-db" >> /opt/DeepEP/BUILD_REF_PARENTS
