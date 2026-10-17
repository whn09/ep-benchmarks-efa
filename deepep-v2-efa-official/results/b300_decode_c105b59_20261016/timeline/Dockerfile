# Experiment-only: per-block phase timeline (EP_TIMELINE) on top of a chcap image.
# timeline.cuh is installed with EP_TIMELINE defined, so this image always records;
# images without this layer do not contain the header at all.
ARG BASE
FROM ${BASE}
COPY timeline.cuh apply_tl.py /tmp/tl/
RUN set -eux; cd /opt/DeepEP; \
    { echo '#define EP_TIMELINE 1'; cat /tmp/tl/timeline.cuh; } > deep_ep/include/deep_ep/common/timeline.cuh; \
    python3 /tmp/tl/apply_tl.py /opt/DeepEP
RUN set -eux; cd /opt/DeepEP; \
    SITE="$(cd / && python3 -c 'import importlib.util,os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("deep_ep").origin)))')"; \
    pip uninstall -y deep-ep deep_ep 2>/dev/null || true; rm -rf "$SITE"/deep_ep*; \
    if [ -f "$SITE/easy-install.pth" ]; then grep -v deep_ep "$SITE/easy-install.pth" > /tmp/pth || true; mv /tmp/pth "$SITE/easy-install.pth"; fi; \
    NCCL_LIB="$(python3 -c 'import nvidia.nccl, os; print(os.path.join(list(nvidia.nccl.__path__)[0], "lib"))')"; \
    LIBRARY_PATH="$NCCL_LIB:/usr/local/cuda/lib64/stubs" python3 setup.py install; rm -rf build; \
    PKG="$(cd / && python3 -c 'import importlib.util,os; print(os.path.dirname(importlib.util.find_spec("deep_ep").origin))')"; \
    grep -q '#define EP_TIMELINE 1' "$(find "$PKG" -name timeline.cuh | head -1)"; \
    grep -q 'EP_TL_MARK(4)' "$(find "$PKG" -name hybrid_dispatch_unordered.cuh | head -1)"; \
    grep -q 'EP_TL_MARK_LANE(5)' "$(find "$PKG" -name hybrid_combine_unordered.cuh | head -1)"; \
    echo "timeline" >> /opt/DeepEP/BUILD_REF_PARENTS
