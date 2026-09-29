namespace bitcomp
{

namespace rle
{

// The 3 most significant bits in the control byte are used for :
// - long or short run length (= if the run length fits in 1 or 2 control bytes)
// - Kind of duplicate : Non-duplicate, 00, FF, other
enum
{
  CTRL_NONDUP = 0x00,
  CTRL_LONG = 0x20,
  CTRL_DUP00 = 0x40,
  CTRL_DUPFF = 0x80,
  CTRL_DUP = 0xC0
};

} // namespace rle

} // namespace bitcomp
