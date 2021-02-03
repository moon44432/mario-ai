
-- This is Lua script for Mesen NES emulator. Run this in the emulator.

function is_dead()
  if emu.read(0x000E, emu.memType.cpu) == 0x0B or emu.read(0x0712, emu.memType.cpu) == 1 then
    return 1
  else
    return 0
  end
end

function printMarioInfo()
  if is_dead() == 1 then
    emu.drawString(250, 2, "D", 0xFFFFFF, 0xFF000000, 1)
  end
end

emu.addEventCallback(printMarioInfo, emu.eventType.endFrame)
