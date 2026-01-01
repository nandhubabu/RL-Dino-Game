import pygame
import random
import numpy as np

class DinoGame:
    def __init__(self):
        pygame.init()
        self.width = 600
        self.height = 300
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # Game State
        self.dino_rect = pygame.Rect(50, 220, 40, 60) # x, y, w, h
        self.cacti = []
        self.dino_vel_y = 0
        self.score = 0
        self.game_over = False

    def reset(self):
        self.dino_rect.y = 220
        self.dino_vel_y = 0
        self.cacti = []
        self.score = 0
        self.game_over = False
        return self.get_state()

    def step(self, action):
        # Action: 0 = Do Nothing, 1 = Jump
        pygame.event.pump()
        
        # 1. Update Dino Physics
        if action == 1 and self.dino_rect.y == 220: # Jump only if on ground
            self.dino_vel_y = -15
        
        self.dino_rect.y += self.dino_vel_y
        self.dino_vel_y += 1 # Gravity
        
        if self.dino_rect.y >= 220: # Floor collision
            self.dino_rect.y = 220
            self.dino_vel_y = 0

        # 2. Update Cacti
        if len(self.cacti) == 0 or self.cacti[-1].x < 400: # Spawn new cactus
            if random.randint(0, 50) == 0:
                self.cacti.append(pygame.Rect(600, 230, 40, 50))
                
        for cactus in self.cacti:
            cactus.x -= 10 # Move left
            if cactus.colliderect(self.dino_rect):
                self.game_over = True
        
        # Remove off-screen cacti
        self.cacti = [c for c in self.cacti if c.x > -50]

        # 3. Draw Everything
        self.screen.fill((255, 255, 255)) # White background
        pygame.draw.rect(self.screen, (0, 0, 0), self.dino_rect) # Black Dino
        for cactus in self.cacti:
            pygame.draw.rect(self.screen, (255, 0, 0), cactus) # Red Cactus
        
        pygame.display.flip()
        self.clock.tick(30) # Limit FPS for viewing (remove for fast training)

        # 4. Return the "Experience"
        # State (Pixels), Reward, Done
        state = self.get_state()
        reward = 0
        if not self.game_over:
            reward = 1       # +1 for staying alive
            if action == 1:  # If it chose to JUMP
                reward -= 5  # -5 penalty for wasting energy!
        else:
            reward = -100    # -100 for dying
        
        return state, reward, self.game_over

    def get_state(self):
        # Capture the raw screen pixels
        # Returns a 3D numpy array (Width, Height, Color)
        view = pygame.surfarray.array3d(self.screen)
        
        # Transpose to (Height, Width, Color) for standard processing
        view = view.transpose([1, 0, 2])
        return view