-- Turns a reference to a label on another page into a link, using this post's cross-references.json.
local index = nil
local function here() return pandoc.path.directory(quarto.doc.input_file) end
local function load()
  if index then return index end
  local f = io.open(pandoc.path.join({here(), "cross-references.json"}), "r")
  if not f then error("cross-references.json not found") end
  local text = f:read("a"); f:close()
  index = quarto.json.decode(text)
  return index
end
function Cite(el)
  local labels = load()
  local post = pandoc.path.filename(here())
  local page = pandoc.path.split_extension(pandoc.path.filename(quarto.doc.input_file)) .. ".html"
  local out = pandoc.Inlines({})
  local changed = false
  for i, c in ipairs(el.citations) do
    local target = labels[c.id]
    if i > 1 then out:insert(pandoc.Str(";")); out:insert(pandoc.Space()) end
    if target and not (target.post == post and target.page == page) then
      changed = true
      local href = (target.post == post and "" or ("../" .. target.post .. "/")) .. target.page .. "#" .. c.id
      out:insert(pandoc.Link(pandoc.Inlines({pandoc.Str(target.text)}), href))
    else
      out:insert(pandoc.Cite(pandoc.Inlines({pandoc.Str("@" .. c.id)}), {c}))
    end
  end
  if changed then return out end
end
