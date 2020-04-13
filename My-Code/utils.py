from skimage.transform import resize
import gym
import torch

def get_screen(env, device, h=80, w=80):
	"""
	Returns a screen render of the environment as torch Tensor.

	Parameters:
		env : gym environment object
		h : height of render
		w : width of render
		device : store location of the render

	Returns:
		k : tensor with screen render
	"""

	k = env.render(mode='rgb_array')
	k = resize(k, (h, w), anti_aliasing=False)
	k = torch.Tensor(k)
	k = k.permute(2,1,0).unsqueeze(0)
	k = k.to(device)
	return k

