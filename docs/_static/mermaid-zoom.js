// Inline pan/zoom for sphinxcontrib-mermaid SVGs.
// - wraps the SVG content in a single <g> and manipulates its transform
// - wheel zoom (centered on cursor), mouse drag pan
// - small toolbar with "+", "-", "↺" buttons positioned at upper-LEFT
//   so it does not collide with the upstream "⛶" fullscreen button (top-right)
(function () {
    "use strict";

    var STATE = new WeakMap();

    function applyTransform(s) {
        s.wrapper.setAttribute(
            "transform",
            "translate(" + s.tx + "," + s.ty + ") scale(" + s.scale + ")"
        );
    }

    function clientToSvg(svg, cx, cy) {
        var pt = svg.createSVGPoint();
        pt.x = cx; pt.y = cy;
        var ctm = svg.getScreenCTM();
        if (!ctm) return { x: cx, y: cy };
        return pt.matrixTransform(ctm.inverse());
    }

    function zoomAt(s, clientX, clientY, factor) {
        var p = clientToSvg(s.svg, clientX, clientY);
        s.tx = p.x - (p.x - s.tx) * factor;
        s.ty = p.y - (p.y - s.ty) * factor;
        s.scale = s.scale * factor;
        applyTransform(s);
    }

    function init(svg) {
        if (STATE.has(svg)) return;
        // Skip if the SVG hasn't finished rendering yet (no children).
        if (!svg.firstChild) return;

        var wrapper = document.createElementNS("http://www.w3.org/2000/svg", "g");
        wrapper.setAttribute("class", "mermaid-zoom-wrapper");
        while (svg.firstChild) wrapper.appendChild(svg.firstChild);
        svg.appendChild(wrapper);

        var s = { svg: svg, wrapper: wrapper, tx: 0, ty: 0, scale: 1 };
        STATE.set(svg, s);

        // Wheel zoom (preventDefault so the page doesn't scroll while zooming).
        svg.addEventListener("wheel", function (e) {
            e.preventDefault();
            zoomAt(s, e.clientX, e.clientY, e.deltaY < 0 ? 1.1 : 1 / 1.1);
        }, { passive: false });

        // Drag pan.
        var dragging = false, lastX = 0, lastY = 0;
        svg.addEventListener("mousedown", function (e) {
            if (e.button !== 0) return;
            dragging = true;
            lastX = e.clientX; lastY = e.clientY;
            svg.style.cursor = "grabbing";
            e.preventDefault();
        });
        function onMove(e) {
            if (!dragging) return;
            var ctm = svg.getScreenCTM();
            if (!ctm) return;
            var dx = (e.clientX - lastX) / ctm.a;
            var dy = (e.clientY - lastY) / ctm.d;
            s.tx += dx; s.ty += dy;
            lastX = e.clientX; lastY = e.clientY;
            applyTransform(s);
        }
        function onUp() {
            if (!dragging) return;
            dragging = false;
            svg.style.cursor = "grab";
        }
        document.addEventListener("mousemove", onMove);
        document.addEventListener("mouseup", onUp);

        svg.style.cursor = "grab";

        // Inject toolbar at upper-LEFT of the surrounding pre/div.mermaid container.
        var box = svg.closest("pre.mermaid, div.mermaid");
        if (!box || box.querySelector(".mermaid-zoom-toolbar")) return;
        if (getComputedStyle(box).position === "static") {
            box.style.position = "relative";
        }

        var tb = document.createElement("div");
        tb.className = "mermaid-zoom-toolbar";
        tb.innerHTML =
            '<button type="button" data-act="in"  title="Zoom in"  aria-label="Zoom in">+</button>' +
            '<button type="button" data-act="out" title="Zoom out" aria-label="Zoom out">−</button>' +
            '<button type="button" data-act="reset" title="Reset"   aria-label="Reset zoom">↺</button>';
        box.appendChild(tb);

        tb.addEventListener("click", function (e) {
            var b = e.target.closest("button");
            if (!b) return;
            var act = b.dataset.act;
            var r = svg.getBoundingClientRect();
            var cx = r.left + r.width / 2;
            var cy = r.top + r.height / 2;
            if (act === "in")    zoomAt(s, cx, cy, 1.25);
            if (act === "out")   zoomAt(s, cx, cy, 1 / 1.25);
            if (act === "reset") {
                s.tx = 0; s.ty = 0; s.scale = 1;
                applyTransform(s);
            }
        });

        // Stop button clicks from triggering the SVG drag.
        tb.addEventListener("mousedown", function (e) { e.stopPropagation(); });
    }

    function scan() {
        // Only attach once mermaid has finished rendering — it flags the
        // container with data-processed="true" when done.
        var nodes = document.querySelectorAll(
            "pre.mermaid[data-processed='true'] svg, " +
            "div.mermaid[data-processed='true'] svg"
        );
        nodes.forEach(init);
    }

    function start() {
        scan();
        if (window.MutationObserver) {
            var obs = new MutationObserver(function () { scan(); });
            obs.observe(document.body, { childList: true, subtree: true });
        }
        var attempts = 0;
        var poll = setInterval(function () {
            scan();
            if (++attempts >= 80) clearInterval(poll);
        }, 200);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", start);
    } else {
        start();
    }
})();
