import jinja2
from PIL import Image

def render_prompt_from_template(template: str, **kwargs):
	with open(template, "r") as f_template:
		template_content = f_template.read()

	rendered_prompt = jinja2.Template(template_content).render(**kwargs)
	print(rendered_prompt)

	return rendered_prompt


def save_image(image: Image.Image, filename: str):
	image.save(filename)
