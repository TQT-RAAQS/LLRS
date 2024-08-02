def xticks(ax, lst):
    ax.set_xticks(lst)
    diff = (lst[-1] - lst[0]) * 0.05
    ax.set_xlim(lst[0] - diff, lst[-1] + diff)
def yticks(ax, lst):
    ax.set_yticks(lst)
    diff = (lst[-1] - lst[0]) * 0.05
    ax.set_ylim(lst[0] - diff, lst[-1] + diff)



def mark(lst):
  ret = []
  for i in lst:
    if i == "lincpu":
      ret.append("8")
    elif i == "lingpu":
      ret.append("<")
    elif i == "bird":
      ret.append("o")
    elif i == "aro":
      ret.append("s")
    elif i == "rrccpu":
      ret.append("^")
    elif i == "rrcgpu":
      ret.append("v")
    elif i == "rrcbatch":
      ret.append(">")
  return ret

def pal(lst):
  prim = []
  sec = []
  for i in lst:
    if i == "lincpu":
      prim.append("#D3D3D3")
      sec.append("#A9A9A9")
    elif i == "lingpu":
      prim.append("#F6B6C7")
      sec.append("#DA1B4E")
    elif i == "bird":
      prim.append("#D9E3FD")
      sec.append("#648EF7")
    elif i == "aro":
      prim.append("#EDD2EE")
      sec.append("#C46BC7")
    elif i == "rrccpu":
      prim.append("#FFF1AD")
      sec.append("#E0BB00")
    elif i == "rrcgpu":
      prim.append("#DAF3CE")
      sec.append("#57B52C")
    elif i == "rrcbatch":
      prim.append("#FFD3AD")
      sec.append("#E06900")
  ret = {'primary': prim, 'secondary': sec}
  return ret

