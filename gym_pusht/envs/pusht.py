import collections

import cv2
import gymnasium as gym
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
import shapely.geometry as sg
from gymnasium import spaces
from pymunk.vec2d import Vec2d

from .pymunk_override import DrawOptions


def pymunk_to_shapely(body, shapes):
    geoms = []
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f"Unsupported shape type {type(shape)}")
    geom = sg.MultiPolygon(geoms)
    return geom


class PushTEnv(gym.Env):
    """
    ## Description

    PushT environment.

    The goal of the agent is to push the block to the goal zone. The agent is a circle and the block is a tee shape.

    ## Action Space

    The action space is continuous and consists of two values: [x, y]. The values are in the range [0, 512] and
    represent the target position of the agent.

    ## Observation Space

    If `obs_type` is set to `state`, the observation space is a 5-dimensional vector representing the state of the
    environment: [agent_x, agent_y, block_x, block_y, block_angle]. The values are in the range [0, 512] for the agent
    and block positions and [0, 2*pi] for the block angle.

    If `obs_type` is set to `pixels`, the observation space is a 96x96 RGB image of the environment.

    ## Rewards

    The reward is the coverage of the block in the goal zone. The reward is 1.0 if the block is fully in the goal zone.

    ## Success Criteria

    The environment is considered solved if the block is at least 95% in the goal zone.

    ## Starting State

    The agent starts at a random position and the block starts at a random position and angle.

    ## Episode Termination

    The episode terminates when the block is at least 95% in the goal zone.

    ## Arguments

    ```python
    >>> import gymnasium as gym
    >>> import gym_pusht
    >>> env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<PushTEnv<gym_pusht/PushT-v0>>>>>
    ```

    * `obs_type`: (str) The observation type. Can be either `state` or `pixels`. Default is `state`.

    * `block_cog`: (tuple) The center of gravity of the block if different from the center of mass. Default is `None`.

    * `damping`: (float) The damping factor of the environment if different from 0. Default is `None`.

    * `render_action`: (bool) Whether to render the action on the image. Default is `True`.

    * `render_size`: (int) The size of the rendered image. Default is `96`.

    * `render_mode`: (str) The rendering mode. Can be either `human` or `rgb_array`. Default is `None`.

    ## Reset Arguments

    Passing the option `options["reset_to_state"]` will reset the environment to a specific state.

    > [!WARNING]  
    > For legacy compatibility, the inner fonctionning has been preserved, and the state set is not the same as the
    > the one passed in the argument.

    ```python
    >>> import gymnasium as gym
    >>> import gym_pusht
    >>> env = gym.make("gym_pusht/PushT-v0")
    >>> state, _ = env.reset(options={"reset_to_state": [0.0, 10.0, 20.0, 30.0, 1.0]})
    >>> state
    array([ 0.      , 10.      , 57.866196, 50.686398,  1.      ],
          dtype=float32)
    ```

    ## Version History

    * v0: Original version

    ## References

    * TODO:
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        obs_type="state",
        block_cog=None,
        damping=None,
        render_action=True,
        render_size=96,
        render_mode=None,
    ):
        super().__init__()
        self.obs_type = obs_type
        if self.obs_type == "state":
            # [agent_x, agent_y, block_x, block_y, block_angle]
            self.observation_space = spaces.Box(
                low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
                high=np.array([512, 512, 512, 512, 2 * np.pi], dtype=np.float32),
            )
        elif self.obs_type == "pixels":
            self.observation_space = spaces.Box(low=0, high=255, shape=(render_size, render_size, 3), dtype=np.uint8)

        self.action_space = spaces.Box(low=0, high=512, shape=(2,), dtype=np.float32)

        # Physics
        self.k_p, self.k_v = 100, 20  # PD control.z
        self.control_hz = self.metadata["render_fps"]
        self.dt = 0.01
        self.block_cog = block_cog
        self.damping = damping

        # Rendering
        self.render_mode = render_mode
        self.render_size = render_size
        self.render_action = render_action

        # If human-rendering is used, `self.window` will be a reference
        # to the window that we draw to. `self.clock` will be a clock that is used
        # to ensure that the environment is rendered at the correct framerate in
        # human-mode. They will remain `None` until human-mode is used for the
        # first time.
        self.window = None
        self.clock = None

        self.teleop = None
        self._last_action = None

        self.success_threshold = 0.95  # 95% coverage

    def step(self, action):
        self.n_contact_points = 0
        n_steps = int(1 / (self.dt * self.control_hz))
        self._last_action = action
        for _ in range(n_steps):
            # Step PD control
            # self.agent.velocity = self.k_p * (act - self.agent.position)    # P control works too.
            acceleration = self.k_p * (action - self.agent.position) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
            self.agent.velocity += acceleration * self.dt

            # Step physics
            self.space.step(self.dt)

            if self.render_mode == "human":
                self.render()

        # Compute reward
        goal_body = self._get_goal_pose_body(self.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)
        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage = intersection_area / goal_area
        reward = np.clip(coverage / self.success_threshold, 0.0, 1.0)
        is_success = coverage > self.success_threshold

        observation = self._get_obs()
        info = self._get_info()

        info["is_success"] = is_success
        terminated = is_success

        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._setup()

        if options is not None and options.get("reset_to_state") is not None:
            state = np.array(options.get("reset_to_state"))
        else:
            state = self.np_random.uniform(low=[50, 50, 100, 100, -np.pi], high=[450, 450, 400, 400, np.pi])
        self._set_state(state)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def _draw(self):
        # Create a screen
        screen = pygame.Surface((512, 512))
        screen.fill((255, 255, 255))
        draw_options = DrawOptions(screen)

        # Draw goal pose
        goal_body = self._get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            goal_points = [goal_body.local_to_world(v) for v in shape.get_vertices()]
            goal_points = [pymunk.pygame_util.to_pygame(point, draw_options.surface) for point in goal_points]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(screen, pygame.Color("LightGreen"), goal_points)

        # Draw agent and block
        self.space.debug_draw(draw_options)
        return screen

    def _get_img(self, screen):
        img = np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))
        img = cv2.resize(img, (self.render_size, self.render_size))
        if self.render_action and self._last_action is not None:
            action = np.array(self._last_action)
            coord = (action / 512 * 96).astype(np.int32)
            marker_size = int(8 / 96 * self.render_size)
            thickness = int(1 / 96 * self.render_size)
            cv2.drawMarker(
                img,
                coord,
                color=(255, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=marker_size,
                thickness=thickness,
            )
        return img

    def render(self):
        screen = self._draw()  # draw the environment on a screen

        if self.render_mode == "rgb_array":
            return self._get_img(screen)
        elif self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((512, 512))
            if self.clock is None:
                self.clock = pygame.time.Clock()

            self.window.blit(screen, screen.get_rect())  # copy our drawings from `screen` to the visible window
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"] * int(1 / (self.dt * self.control_hz)))
            pygame.display.update()
        else:
            return None

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def teleop_agent(self):
        teleop_agent = collections.namedtuple("TeleopAgent", ["act"])

        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act

        return teleop_agent(act)

    def _get_obs(self):
        if self.obs_type == "state":
            agent_position = np.array(self.agent.position)
            block_position = np.array(self.block.position)
            block_angle = self.block.angle % (2 * np.pi)
            obs = np.concatenate([agent_position, block_position, [block_angle]], dtype=np.float32)
        elif self.obs_type == "pixels":
            screen = self._draw()
            obs = self._get_img(screen)
        return obs

    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        # preserving the legacy assignment order for compatibility
        # the order here doesn't matter somehow, maybe because CoM is aligned with body origin
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body

    def _get_info(self):
        n_steps = int(1 / self.dt * self.control_hz)
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            "pos_agent": np.array(self.agent.position),
            "vel_agent": np.array(self.agent.velocity),
            "block_pose": np.array(list(self.block.position) + [self.block.angle]),
            "goal_pose": self.goal_pose,
            "n_contacts": n_contact_points_per_step,
        }
        return info

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = self.damping if self.damping is not None else 0.0
        self.teleop = False

        # Add walls
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2),
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone
        self.agent = self._add_circle((256, 400), 15)
        self.block = self._add_tee((256, 300), 0)
        self.goal_pose = np.array([256, 256, np.pi / 4])  # x, y, theta (in radians)
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog

        # Add collision handling
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

    def _set_state(self, state):
        self.agent.position = list(state[:2])
        # Setting angle rotates with respect to center of mass, therefore will modify the geometric position if not
        # the same as CoM. Therefore should theoritically set the angle first. But for compatibility with legacy data,
        # we do the opposite.
        self.block.position = list(state[2:4])
        self.block.angle = state[4]

        # Run physics to take effect
        self.space.step(self.dt)

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color("LightGray")  # https://htmlcolorcodes.com/color-names
        return shape

    def _add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color("RoyalBlue")
        self.space.add(body, shape)
        return body

    def _add_tee(self, position, angle, scale=30, color="LightSlateGray", mask=None):
        if mask is None:
            mask = pymunk.ShapeFilter.ALL_MASKS()
        mass = 1
        length = 4
        vertices1 = [
            (-length * scale / 2, scale),
            (length * scale / 2, scale),
            (length * scale / 2, 0),
            (-length * scale / 2, 0),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [
            (-scale / 2, scale),
            (-scale / 2, length * scale),
            (scale / 2, length * scale),
            (scale / 2, scale),
        ]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body