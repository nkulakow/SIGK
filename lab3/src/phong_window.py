import os.path

import moderngl
import numpy as np
from PIL import Image
from pyrr import Matrix44
import json

from base_window import BaseWindow


class PhongWindow(BaseWindow):

    def __init__(self, **kwargs):
        super(PhongWindow, self).__init__(**kwargs)
        self.frame = 0
        self.metadata = [] # New: Store scene parameters

    def init_shaders_variables(self):
        self.model_view_projection = self.program["model_view_projection"]
        self.model_matrix = self.program["model_matrix"]
        self.material_diffuse = self.program["material_diffuse"]
        self.material_shininess = self.program["material_shininess"]
        self.light_position = self.program["light_position"]
        self.camera_position = self.program["camera_position"]

    def on_render(self, time: float, frame_time: float):
        if self.frame >= 3000: # New: Generate exactly 3000 frames
            if self.output_path:
                json_path = os.path.join(self.output_path, 'scene_parameters.json')
                with open(json_path, 'w') as f:
                    json.dump(self.metadata, f, indent=2)
                print(f"Dataset is saved in {json_path}")
            self.wnd.close()
            return
            
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        
        # New: Randomize some scene parameters and store them
        model_translation = np.random.uniform(-20.0, 20.0, 3).tolist()
        material_diffuse = np.random.uniform(0.0, 1.0, 3).tolist() 
        material_shininess = float(np.random.uniform(3.0, 20.0))
        light_position = np.random.uniform(-20.0, 20.0, 3).tolist()
        camera_position = [5.0, 5.0, 15.0]

        model_matrix = Matrix44.from_translation(model_translation)
        proj = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000.0)
        
        lookat = Matrix44.look_at(
            camera_position,
            model_translation, # New: change (0.0, 0.0, 0.0) to model_translation to keep the object in the center of the view
            (0.0, 1.0, 0.0),
        )
        
        model_view_projection = proj * lookat * model_matrix

        self.model_view_projection.write(model_view_projection.astype('f4').tobytes())
        self.model_matrix.write(model_matrix.astype('f4').tobytes())
        self.material_diffuse.write(np.array(material_diffuse, dtype='f4').tobytes())
        self.material_shininess.write(np.array([material_shininess], dtype='f4').tobytes())
        self.light_position.write(np.array(light_position, dtype='f4').tobytes())
        self.camera_position.write(np.array(camera_position, dtype='f4').tobytes())

        self.vao.render()
        if self.output_path:
            img_name = f'image_{self.frame:04}.png'
            img = (
                Image.frombuffer('RGBA', self.wnd.size, self.wnd.fbo.read(components=4))
                     .transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            )

            img.save(os.path.join(self.output_path, img_name))
            
            # New: Calculate features for the network and store metadata
            rel_light_pos = [l - o for l, o in zip(light_position, model_translation)]
            distance = np.linalg.norm(np.array(camera_position) - np.array(model_translation))
            inv_dist = 1.0 / distance if distance > 0 else 0.0
            
            self.metadata.append({
                "image_name": img_name,
                "model_translation": model_translation,
                "material_diffuse": material_diffuse,
                "material_shininess": material_shininess,
                "light_position": light_position,
                "camera_position": camera_position,
                "features_for_network": {
                    "relative_light_position": rel_light_pos,
                    "inverse_depth": inv_dist
                }
            })
            self.frame += 1
