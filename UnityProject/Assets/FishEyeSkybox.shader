Shader "Unlit/FishEyeSkybox"
{
    Properties
    {
        _FrontTex("Front FishEye Texture", 2D) = "white" {}
        _BackTex("Back FishEye Texture", 2D) = "white" {}
        _FOV("FishEye FOV (Degrees)", Float) = 180.0
        _Scale("FishEye Scale", Float) = 1.0
        _Opacity("Opacity", Range(0,1)) = 1.0
        [Toggle]_Grayscale("Use Grayscale", Float) = 0

        // 👇 카메라 위치 및 기준점 위치를 외부에서 설정 가능하도록
        _CameraWorldPos("Camera World Position", Vector) = (0,0,0,0)
        _OriginWorldPos("Origin World Position", Vector) = (0,0,0,0)
    }

    SubShader
    {
        Tags { "Queue"="Transparent" "RenderType"="Transparent" }
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            Blend SrcAlpha OneMinusSrcAlpha
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _FrontTex;
            sampler2D _BackTex;
            float _FOV;
            float _Scale;
            float _Opacity;
            float _Grayscale;

            float4 _CameraWorldPos;
            float4 _OriginWorldPos;

            struct appdata
            {
                float4 vertex : POSITION;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float3 worldDir : TEXCOORD0;
            };

            v2f vert (appdata v)
            {
                v2f o;
                float4 worldPos = mul(unity_ObjectToWorld, v.vertex);
                o.pos = UnityObjectToClipPos(v.vertex);
                o.worldDir = normalize(worldPos.xyz);
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                float3 dir = normalize(i.worldDir);
                float halfFOV = radians(_FOV) * 0.5;

                float frontTheta = acos(dot(dir, float3(0, 0, 1)));
                float frontR = frontTheta / halfFOV;
                float frontPhi = atan2(dir.y, dir.x);
                float2 fishUV_front = float2(0.5, 0.5) + _Scale * float2(cos(frontPhi), sin(frontPhi)) * (frontR * 0.5);

                float backTheta = acos(dot(dir, float3(0, 0, -1)));
                float backR = backTheta / halfFOV;
                float backPhi = atan2(dir.y, dir.x) + 3.14159265;
                float2 fishUV_back = float2(0.5, 0.5) + _Scale * float2(cos(backPhi), sin(backPhi)) * (backR * 0.5);

                bool validFront = (frontR * _Scale <= 1.0);
                bool validBack = (backR * _Scale <= 1.0);

                fixed4 frontColor = tex2D(_FrontTex, fishUV_front);
                fixed4 backColor = tex2D(_BackTex, float2(fishUV_back.x, 1.0 - fishUV_back.y));

                if (_Grayscale > 0.5)
                {
                    float fLum = dot(frontColor.rgb, float3(0.299, 0.587, 0.114));
                    frontColor.rgb = float3(fLum, fLum, fLum);
                    
                    float bLum = dot(backColor.rgb, float3(0.299, 0.587, 0.114));
                    backColor.rgb = float3(bLum, bLum, bLum);
                }

                // 👉 카메라-원점 거리 기반 채도 보정
                float3 camToOrigin = _CameraWorldPos.xyz - _OriginWorldPos.xyz;
                float dist = length(camToOrigin);
                float saturation = saturate(1.0 - dist * 0.1); // 거리 10 이상이면 거의 무채색

                frontColor.rgb = lerp(dot(frontColor.rgb, float3(0.299, 0.587, 0.114)).xxx, frontColor.rgb, saturation);
                backColor.rgb = lerp(dot(backColor.rgb, float3(0.299, 0.587, 0.114)).xxx, backColor.rgb, saturation);

                if (validFront && !validBack)
                    return frontColor * _Opacity;
                else if (!validFront && validBack)
                    return backColor * _Opacity;
                else if (!validFront && !validBack)
                    discard;

                float frontWeight = 1.0 - saturate(frontR * _Scale);
                float backWeight = 1.0 - saturate(backR * _Scale);
                float total = frontWeight + backWeight;
                float blend = (total > 0.0) ? (backWeight / total) : 0.5;

                fixed4 col = lerp(frontColor, backColor, blend);
                col.a *= _Opacity;
                return col;
            }

            ENDCG
        }
    }
    FallBack "Diffuse"
}
