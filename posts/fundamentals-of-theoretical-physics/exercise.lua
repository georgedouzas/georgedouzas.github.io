-- Shortcode {{< sage <label> >}}: shows a Sage solution and its recorded output from this post's sage-exercises folder.
local function slurp(path)
  local f = io.open(path, "r")
  if not f then return nil end
  local s = f:read("a"); f:close(); return s
end
return {
  ["sage"] = function(args, kwargs, meta)
    local label = pandoc.utils.stringify(args[1])
    local dir = pandoc.path.join({pandoc.path.directory(quarto.doc.input_file), "sage-exercises"})
    local src = slurp(pandoc.path.join({dir, label .. ".sage"}))
    if not src then error("sage shortcode: no script " .. label .. ".sage") end
    local out = slurp(pandoc.path.join({dir, label .. ".out"}))
    if not out then error("sage shortcode: no recorded output " .. label .. ".out") end
    local body = src
    while body:sub(1, 1) == "#" do
      local nl = body:find("\n", 1, true)
      if not nl then body = ""; break end
      body = body:sub(nl + 1)
    end
    return pandoc.Div({
      pandoc.CodeBlock(body:gsub("%s+$", ""), pandoc.Attr("", {"python"})),
      pandoc.Para({pandoc.Emph({pandoc.Str("Output:")})}),
      pandoc.CodeBlock(out:gsub("%s+$", ""), pandoc.Attr("", {"text"})),
    }, pandoc.Attr("", {"sage-solution"}))
  end
}
