"""OpenGL visualisation of the SafePath environment (pygame window + PyOpenGL).

Scene: the survivor (glowing orb) travels along a pathway of five stations,
Safe space -> Trust -> Options -> Consent -> Support. Her position is driven by
the true environment state. A translucent shield dome appears when privacy mode
is on, a red pulse when a safety risk is flagged, and the right-hand panel shows
live gauges, the agent's last action and the event log.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pygame
from OpenGL.GL import (
    GL_BLEND, GL_COLOR_BUFFER_BIT, GL_LINE_LOOP, GL_LINE_STRIP, GL_LINES,
    GL_MODELVIEW, GL_ONE_MINUS_SRC_ALPHA, GL_PROJECTION, GL_QUADS, GL_RGBA,
    GL_SRC_ALPHA, GL_TRIANGLE_FAN, GL_UNSIGNED_BYTE, GL_LINE_SMOOTH, GL_RGB,
    glBegin, glBlendFunc, glClear, glClearColor, glColor4f, glDrawPixels,
    glEnable, glEnd, glLineWidth, glLoadIdentity, glMatrixMode, glOrtho,
    glRasterPos2f, glReadPixels, glVertex2f, glPixelStorei, GL_PACK_ALIGNMENT,
)

W, H = 1100, 640
STATIONS = ["Safe space", "Trust", "Options", "Consent", "Support"]
ST_POS = [(90, 420), (230, 300), (380, 380), (520, 260), (660, 340)]
PURPLE = (0.55, 0.36, 0.85)
TEAL = (0.18, 0.75, 0.70)
ROSE = (0.95, 0.35, 0.45)
AMBER = (0.98, 0.72, 0.25)


class SafePathRenderer:
    def __init__(self, offscreen: bool = False):
        if offscreen:
            os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "-2000,-2000")
        pygame.init()
        flags = pygame.OPENGL | pygame.DOUBLEBUF
        if offscreen:
            flags |= pygame.HIDDEN
        pygame.display.set_mode((W, H), flags)
        pygame.display.set_caption("SafePath - support navigation agent (OpenGL)")
        self.font = pygame.font.SysFont("segoeui,arial", 17)
        self.big = pygame.font.SysFont("segoeui,arial", 24, bold=True)
        self.small = pygame.font.SysFont("segoeui,arial", 14)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, W, H, 0, -1, 1)          # y down, like screen coords
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_BLEND)
        glEnable(GL_LINE_SMOOTH)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.offscreen = offscreen
        self.frame = 0
        self.log: list[str] = []
        self.pos = list(ST_POS[0])

    # ---------------------------------------------------------- primitives --
    @staticmethod
    def circle(cx, cy, r, color, a=1.0, seg=48):
        glColor4f(*color, a)
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(cx, cy)
        for i in range(seg + 1):
            t = 2 * math.pi * i / seg
            glVertex2f(cx + r * math.cos(t), cy + r * math.sin(t))
        glEnd()

    @staticmethod
    def ring(cx, cy, r, color, a=1.0, w=2.0, seg=64):
        glLineWidth(w)
        glColor4f(*color, a)
        glBegin(GL_LINE_LOOP)
        for i in range(seg):
            t = 2 * math.pi * i / seg
            glVertex2f(cx + r * math.cos(t), cy + r * math.sin(t))
        glEnd()

    @staticmethod
    def rect(x, y, w, h, color, a=1.0):
        glColor4f(*color, a)
        glBegin(GL_QUADS)
        glVertex2f(x, y); glVertex2f(x + w, y); glVertex2f(x + w, y + h); glVertex2f(x, y + h)
        glEnd()

    def text(self, s, x, y, font=None, color=(235, 235, 245)):
        # pygame renders the text to an image, then the pixels are copied into the OpenGL frame
        surf = (font or self.font).render(s, True, color)
        data = pygame.image.tostring(surf, "RGBA", True)
        glRasterPos2f(x, y + surf.get_height())
        glDrawPixels(surf.get_width(), surf.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, data)

    def gauge(self, label, value, x, y, color):
        self.text(f"{label}", x, y - 2, self.small)
        self.rect(x + 110, y, 200, 14, (0.25, 0.25, 0.32))
        self.rect(x + 110, y, 200 * value, 14, color)
        self.text(f"{value:.2f}", x + 318, y - 2, self.small)

    # --------------------------------------------------------------- draw --
    def _progress_stage(self, env) -> float:
        """Continuous position along the path from the true state."""
        if env.outcome.startswith(("Connected", "At-risk", "Escalated")):
            return 4.0
        stage = 0.0
        stage += min(1.0, env.trust / 0.5)
        stage += min(1.0, env.info / 4 + 0.5 * env.pref_known)
        stage += 1.0 if env.consent else 0.5 * env.readiness
        return min(stage, 3.6)

    def draw(self, env):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
        self.frame += 1
        if env.last_event and (not self.log or self.log[-1] != f"t={env.t:02d} {env.last_event}"):
            self.log.append(f"t={env.t:02d} {env.last_event}")
            self.log = self.log[-9:]

        glClearColor(0.07, 0.06, 0.12, 1)
        glClear(GL_COLOR_BUFFER_BIT)
        # soft background glow
        for i in range(6):
            self.circle(380, 340, 330 - i * 45, PURPLE, 0.03)

        # path
        glLineWidth(6)
        glColor4f(0.45, 0.42, 0.6, 0.6)
        glBegin(GL_LINE_STRIP)
        for p in ST_POS:
            glVertex2f(*p)
        glEnd()

        stage = self._progress_stage(env)
        for i, (p, name) in enumerate(zip(ST_POS, STATIONS)):
            reached = stage >= i - 1e-6
            col = TEAL if reached else (0.4, 0.4, 0.5)
            self.circle(*p, 22, col, 0.9 if reached else 0.5)
            self.ring(*p, 28, col, 0.6)
            self.text(name, p[0] - 32, p[1] + 34, self.small)

        # survivor orb interpolates toward target position
        k = int(min(stage, 3.999))
        f = stage - k
        tx = ST_POS[k][0] + f * (ST_POS[k + 1][0] - ST_POS[k][0])
        ty = ST_POS[k][1] + f * (ST_POS[k + 1][1] - ST_POS[k][1]) - 45
        self.pos[0] += 0.5 * (tx - self.pos[0])
        self.pos[1] += 0.5 * (ty - self.pos[1])
        x, y = self.pos
        pulse = 0.5 + 0.5 * math.sin(self.frame * 0.6)
        # distress = warm colour, calm = cool colour
        d = env.distress
        orb = (0.3 + 0.65 * d, 0.55 - 0.25 * d, 0.95 - 0.5 * d)
        for i in range(5):
            self.circle(x, y, 26 + i * 6, orb, 0.07)
        self.circle(x, y, 18, orb, 1.0)
        if env.privacy_on:
            self.circle(x, y, 44, TEAL, 0.15)
            self.ring(x, y, 44, TEAL, 0.9, 3)
        if env.safety_checked and env.risk:
            self.ring(x, y, 54 + 10 * pulse, ROSE, 0.9, 3)
            self.text("RISK FLAGGED", x - 45, y + 50, self.small, (255, 120, 140))
        if env.shared_device and not env.privacy_on:
            self.text(f"shared device exposure {env.exposure:.2f}", 40, 560, self.small, (255, 200, 120))

        # title
        self.text("SafePath  ·  AI support navigator", 30, 20, self.big)
        self.text("The agent offers; the survivor decides.", 32, 54, self.small, (190, 180, 220))

        # ---- side panel ----
        px = 740
        self.rect(px - 15, 0, W - px + 15, H, (0.11, 0.10, 0.18))
        self.text(f"Step {env.t}/{env.max_steps}", px, 20, self.big)
        from environment.custom_env import ACTIONS, SERVICES
        act = "-" if env.last_action is None else ACTIONS[env.last_action]
        self.text(f"Agent action: {act}", px, 60, self.font, (250, 210, 120))
        self.gauge("Distress", env.distress, px, 100, ROSE)
        self.gauge("Trust", env.trust, px, 126, TEAL)
        self.gauge("Readiness", env.readiness, px, 152, PURPLE)
        self.gauge("Info coverage", env.info / 4, px, 178, AMBER)
        self.gauge("Pressure", env.pressure, px, 204, (0.9, 0.5, 0.3))
        flags = [
            ("Consent", env.consent), ("Privacy on", env.privacy_on),
            ("Safety checked", env.safety_checked), ("Preference known", env.pref_known),
        ]
        for i, (n, v) in enumerate(flags):
            cx = px + (i % 2) * 175
            cy = 240 + (i // 2) * 24
            self.circle(cx + 6, cy + 9, 6, TEAL if v else (0.35, 0.35, 0.42))
            self.text(n, cx + 18, cy, self.small)
        pref = SERVICES[env.preference] if env.pref_known else "unknown"
        self.text(f"Wants: {pref}", px, 292, self.small, (200, 200, 230))
        self.text("Event log", px, 322, self.font, (190, 180, 220))
        for i, line in enumerate(self.log):
            self.text(line[:44], px, 348 + i * 22, self.small)
        oc = env.outcome
        col = (130, 240, 170) if oc.startswith(("Connected", "At-risk")) else \
              (255, 130, 140) if oc.startswith(("PRIVACY", "She left")) else (220, 220, 230)
        self.text(f"Outcome: {oc}", 30, 600, self.font, col)

        if self.offscreen:
            glPixelStorei(GL_PACK_ALIGNMENT, 1)
            buf = glReadPixels(0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE)
            img = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3)[::-1]
            return img.copy()
        pygame.display.flip()
        return None

    def close(self):
        pygame.display.quit()
        pygame.quit()
